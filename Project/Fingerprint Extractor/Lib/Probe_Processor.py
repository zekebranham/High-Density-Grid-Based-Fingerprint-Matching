from Lib.Database_Load_Processor import process_fingerprint_image
from Lib.database_worker import connect_db, create_tables, insert_fingerprint, get_next_fingerprint_id, insert_data
import os
import json

def probe_processor(image_path):
    triplets_data = process_fingerprint_image(image_path)
    return triplets_data

def probe_dataBuilder(triplets_data):
    probe_data = {}
    for grid_label, data in triplets_data.items():
        probe_data[grid_label] = []
        for triplet in data["Triplets"]:
            minutiae_points = json.dumps(triplet['MinutiaePoints'])
            distances = json.dumps(triplet['Distances'])
            angles = json.dumps(triplet['Angles'])
            probe_data[grid_label].append({
                'MinutiaePoints': minutiae_points,
                'Distances': distances,
                'Angles': angles
            })
    return probe_data