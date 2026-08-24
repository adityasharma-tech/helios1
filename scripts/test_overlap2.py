from pathlib import Path
from match_coordinates import parse_isro_xml, parse_jaxa_lbl
import os

isro_xml_path = "/media/friday/Toshiba Drive/FILES/helios/ISRO/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260103/data/calibrated/20260103/ch2_ohr_ncp_20260103T0410224157_d_img_d18.xml"
isro_bounds = parse_isro_xml(isro_xml_path)

jaxa_dir = "/home/friday/helios1/data/selene_overlapping_data"
jaxa_lbls = list(Path(jaxa_dir).rglob('*_img.lbl'))

for lbl in jaxa_lbls:
    b = parse_jaxa_lbl(lbl)
    if b:
        is_overlapping = not (b['max_lon'] < isro_bounds['min_lon'] or b['min_lon'] > isro_bounds['max_lon'] or b['max_lat'] < isro_bounds['min_lat'] or b['min_lat'] > isro_bounds['max_lat'])
        if is_overlapping:
            img_path = str(lbl).replace('.lbl', '.img')
            if os.path.exists(img_path) and os.path.getsize(img_path) > 1000000:
                print(lbl.name, b)

