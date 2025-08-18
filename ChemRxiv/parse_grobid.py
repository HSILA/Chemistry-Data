import numpy as np
from cached_path import cached_path
from blingfire import text_to_words
from typing import List, Union
import argparse


import os
import re
import xml.etree.ElementTree as ET
import tqdm
import unicodedata
import datasets
import json
import shutil
from typing import Optional


GOOGLE_1T_CORPUS = "https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/lucas/google-1T-unigram/unigram_freq.csv"


def preprocess_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"https?://\S+", "", text)
    pattern = r"\s*ORCID:\s*.*?Content not peer-reviewed by ChemRxiv\. License:\s*CC BY(?:-[A-Za-z]+)*\s*4\.0"
    text = re.sub(pattern, "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_text_without_refs(elem):
    """
    Recursively extract <p> tags from an element but skip the text within <ref> tags.
    """
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        # Check if the child's local tag is 'ref'
        if child.tag.split("}")[-1] == "ref":
            # Skip its text, but keep its tail if available.
            if child.tail:
                parts.append(child.tail)
        else:
            parts.append(get_text_without_refs(child))
            if child.tail:
                parts.append(child.tail)
    return "".join(parts).strip()


def extract_paragraphs(xml_file_path: str, concat_p: bool):
    """
    Extracts the text for each <div> in the <body> of a TEI XML file (Grobid output).
    For each <div>, all <p> tags are located, their text is extracted using
    `get_text_without_refs` to remove reference text. Depending on the `concat_p`
    flag, the paragraphs are either concatenated into a single text for each <div>
    or kept as separate entries.

    Args:
        xml_file_path (str): Path to the XML file.
        concat_p (bool): If True, concatenate all paragraphs within a <div> into
                         a single text. If False, keep each paragraph separate.

    Returns:
        list of str: A list of strings, each representing either the concatenated
                     text from one <div> element or individual paragraphs.
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    # Find the <body> element.
    body = root.find(".//tei:body", ns)
    if body is None:
        return []

    # Find all <div> elements within the <body>.
    divs = body.findall(".//tei:div", ns)
    div_texts = []

    for div in divs:
        # For each div, get all <p> elements.
        p_elements = div.findall(".//tei:p", ns)
        p_texts = []
        for p in p_elements:
            # Use get_text_without_refs to clean the <p> text.
            cleaned_text = get_text_without_refs(p)
            if cleaned_text:
                p_texts.append(cleaned_text)

        if concat_p:
            # Concatenate all paragraphs for this div into one standalone text.
            div_text = "\n".join(p_texts).strip()
            if div_text:
                div_texts.append(div_text)
        else:
            # Add each paragraph as a separate element in div_texts.
            div_texts.extend(p_texts)

    return div_texts


class UnigramPerplexityPredictor:
    """Predicts the perplexity of a passage based on the unigram distribution
    probability of the words in a large corpus."""

    UNK = "<unk>"

    def __init__(self, word_counts_path: str = GOOGLE_1T_CORPUS):
        local_word_counts_path = cached_path(word_counts_path)
        with open(local_word_counts_path) as f:
            word_counts = {
                word: int(count)
                for word, count in (line.strip().split(",", 1) for line in f)
                if count.isnumeric()
            }

        word_total = sum(word_counts.values())
        word_total_log = np.log2(word_total)
        self.words_logp = {
            word: np.log2(count) - word_total_log for word, count in word_counts.items()
        }

        # <unk> token has fictional count of √vocab_size + 1
        self.words_logp[self.UNK] = (
            np.log2(np.sqrt(len(self.words_logp)) + 1) - word_total_log
        )

    def log_p(self, word: str) -> float:
        return self.words_logp.get(word.lower(), self.words_logp[self.UNK])

    def predict(self, text: Union[str, List[str]]) -> float:
        if isinstance(text, str):
            text = text_to_words(text).split()

        log_prob = sum(self.log_p(word) / len(text) for word in text)
        return log_prob


def xml_to_text(
    src_path: str,
    dst_path: str,
    hf_dataset_path: Optional[str] = None,
    hf_config: Optional[str] = None,
    concat_p: bool = True,
):
    """
    Converts XML files from a source directory into JSON files containing
    filtered paragraphs and optionally pushes the results to the Hugging Face Hub.

    This function processes each XML file in the specified source directory,
    extracts paragraphs, preprocesses them, and filters them based on length
    and unigram perplexity. The filtered paragraphs are then saved as JSON
    files in the destination directory. If a Hugging Face dataset path is
    provided, the results are also pushed to the Hugging Face Hub.

    Args:
        src_path (str): The source directory containing XML files.
        dst_path (str): The destination directory to save JSON files.
        hf_dataset_path (Optional[str]): The Hugging Face dataset path to push
            the results. Defaults to None.
        hf_config (Optional[str]): The Hugging Face dataset config name.
            Defaults to None.
        concat_p (bool): If True, concatenate all paragraphs within a <div> into
                         a single text. If False, keep each paragraph separate.

    Returns:
        None
    """
    os.makedirs(dst_path, exist_ok=True)

    xlm_files = [f for f in os.listdir(src_path) if f.endswith(".xml")]

    upp = UnigramPerplexityPredictor()

    results = []
    num_filtered = 0
    total_ps = 0

    for file in tqdm.tqdm(xlm_files):
        xml_path = os.path.join(src_path, file)
        id, _ = os.path.splitext(file)
        id = id.split(".")[0] if "." in id else id
        json_output_path = os.path.join(dst_path, f"{id}-paragraphs.json")

        paragraphs = extract_paragraphs(xml_path, concat_p)
        len_p_before = len(paragraphs)
        paragraphs = [preprocess_text(p) for p in paragraphs]
        paragraphs = [p for p in paragraphs if len(p.split()) > 50]
        paragraphs = [p for p in paragraphs if upp.predict(p) > -20]
        num_filtered += len_p_before - len(paragraphs)
        total_ps += len(paragraphs)

        for idx, paragraph in enumerate(paragraphs):
            results.append({"id": id, "idx": idx, "paragraph": paragraph})

        with open(json_output_path, "w", encoding="utf-8") as json_file:
            json.dump(paragraphs, json_file, ensure_ascii=False, indent=4)

    print(f"#Filtered Paragraphs: {num_filtered}\t# Total Paragraphs: {total_ps}")

    if hf_dataset_path:
        ds = datasets.Dataset.from_list(results)
        ds.push_to_hub(hf_dataset_path, config_name=hf_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert XML files to text and optionally push to Hugging Face Hub."
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Source directory containing XML files.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="Destination directory to save JSON files.",
    )
    parser.add_argument(
        "--hf-path",
        type=str,
        default=None,
        help="Hugging Face dataset path to push the results.",
    )
    parser.add_argument(
        "--hf-config",
        type=str,
        default=None,
        help="Hugging Face dataset config name.",
    )
    parser.add_argument(
        "--concat-p",
        action="store_true",
        help="Concatenate all paragraphs within a <div> into a single text.",
    )

    args = parser.parse_args()

    xml_to_text(
        src_path=args.src,
        dst_path=args.dst,
        hf_dataset_path=args.hf_path,
        hf_config=args.hf_config,
        concat_p=args.concat_p,
    )
