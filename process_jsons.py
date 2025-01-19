import os
import json
import pandas as pd
import argparse
import tqdm
import re


def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def read_path_jsons(directory):
    for file in sorted(os.listdir(directory), key=natural_sort_key):
        if file.endswith(".json"):
            yield read_json(os.path.join(directory, file))


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', s)]


def append_to_csv(df, csv_path):
    with open(csv_path, 'a') as f:
        df.to_csv(f, header=f.tell() == 0, index=False)


def get_references(data):
    refs = {}
    for r in data['Record']['Reference']:
        refs[r['ReferenceNumber']] = r
    return refs


def get_section(sections, key):
    for sec in sections:
        if sec['TOCHeading'] == key:
            return sec
    return None


def get_descriptions(names_identifiers_section):
    ref_desc = {}
    records_desc = get_section(names_identifiers_section['Section'], 'Record Description')
    if records_desc is None:
        return ref_desc
    for item in records_desc['Information']:
        ref_number = item['ReferenceNumber']
        if ref_number != 111:
            ref_desc[ref_number] = item['Value']['StringWithMarkup'][0]['String']
    return ref_desc


def get_descriptors(names_identifiers_section):
    computed_descriptors = get_section(names_identifiers_section['Section'], 'Computed Descriptors')
    iupac_name = smiles = inchi = None
    for item in computed_descriptors['Section']:
        if item['TOCHeading'] == 'IUPAC Name':
            iupac_name = item['Information'][0]['Value']['StringWithMarkup'][0]['String']
        elif item['TOCHeading'] == 'InChI':
            inchi = item['Information'][0]['Value']['StringWithMarkup'][0]['String']
        elif item['TOCHeading'] == 'SMILES':
            smiles = item['Information'][0]['Value']['StringWithMarkup'][0]['String']
    return iupac_name, smiles, inchi


def get_molecular_formula(names_identifiers_section):
    molecular_formula = get_section(names_identifiers_section['Section'], 'Molecular Formula')
    return molecular_formula['Information'][0]['Value']['StringWithMarkup'][0]['String']


def get_synonyms(names_identifiers_section):
    try:
        synonyms_sec = get_section(names_identifiers_section['Section'], 'Synonyms')
        synonyms = get_section(synonyms_sec['Section'], 'Depositor-Supplied Synonyms')
        syns_list = synonyms['Information'][0]['Value']['StringWithMarkup']
        return [syn['String'] for syn in syns_list]
    except:
        return []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process JSON files from a directory.')
    parser.add_argument('--jsons-dir', type=str, help='Directory containing JSON files')
    parser.add_argument('--comp-csv', type=str, help='CSV file to save compounds data')
    parser.add_argument('--desc-csv', type=str, help='CSV file to save descriptions data')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Number of JSON files to process in a batch')
    args = parser.parse_args()

    comp_rows, desc_rows = [], []
    json_files = read_path_jsons(args.jsons_dir)

    num_jsons = len(os.listdir(args.jsons_dir))

    for i, data in enumerate(tqdm.tqdm(json_files, total=num_jsons)):
        if i % args.batch_size == 0 and i != 0:
            append_to_csv(pd.DataFrame(comp_rows), args.comp_csv)
            append_to_csv(pd.DataFrame(desc_rows), args.desc_csv)
            comp_rows, desc_rows = [], []

        id = data['Record']['RecordNumber']
        id_type = data['Record']['RecordType']
        title = data['Record']['RecordTitle']

        references = get_references(data)
        main_sections = data['Record']['Section']

        name_ids_secion = get_section(main_sections, 'Names and Identifiers')
        descriptions = get_descriptions(name_ids_secion)

        iupac_name, smiles, inchi = get_descriptors(name_ids_secion)
        molecular_formula = get_molecular_formula(name_ids_secion)
        synonyms = get_synonyms(name_ids_secion)

        comp_row = {
            'CID': id,
            'Title': title,
            'MolecularFormula': molecular_formula,
            'IUPACName': iupac_name,
            'InChI': inchi,
            'SMILES': smiles,
            'Synonyms': synonyms
        }
        comp_rows.append(comp_row)

        if descriptions:
            for ref_id, desc in descriptions.items():
                ref = references[ref_id]

                desc_row = {
                    'CID': id,
                    'Title': title,
                    'Description': desc,
                    'ReferenceNumber': ref_id,
                    'SourceName': ref['SourceName'],
                    'SourceID': ref['SourceID'],
                    'ReferenceDescription': ref['Description'],
                    'URL': ref['URL']
                }

                desc_rows.append(desc_row)

    # Flush remaining rows
    if comp_rows:
        append_to_csv(pd.DataFrame(comp_rows), args.comp_csv)
    if desc_rows:
        append_to_csv(pd.DataFrame(desc_rows), args.desc_csv)
