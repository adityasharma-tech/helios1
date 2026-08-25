import os
from pathlib import Path
from match_coordinates import parse_isro_xml, parse_jaxa_lbl, check_overlap

def main():
    isro_dir = "/media/friday/Toshiba Drive/FILES/helios/ISRO/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/"
    jaxa_dir = "/home/friday/helios1/data/selene_metadata"
    
    isro_files = list(Path(isro_dir).rglob('*.xml'))
    jaxa_files = list(Path(jaxa_dir).rglob('*.lbl'))
    
    isro_bounds = [b for f in isro_files if (b := parse_isro_xml(f)) is not None]
    jaxa_bounds = [b for f in jaxa_files if (b := parse_jaxa_lbl(f)) is not None]
    
    match_count = 0
    for isro in isro_bounds:
        for jaxa in jaxa_bounds:
            if check_overlap(isro, jaxa):
                match_count += 1
                print(f"[ MATCH {match_count} ]")
                print(f"ISRO OHRC: {Path(isro['file']).name}")
                print(f"JAXA SELENE: {Path(jaxa['file']).name}\n")
                
if __name__ == '__main__':
    main()
