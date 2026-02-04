import pandas as pd

target_dict_mapping = {
    'target-hemoglobin-grade2plus': 'Hb',
    'target-neutrophil-grade2plus': 'ANC',
    'target-bilirubin-grade2plus': 'Bili',
    'target-platelet-grade2plus': 'PLT',
    'target-ALT-grade2plus': 'ALT',
    'target-AST-grade2plus': 'AST',
    'target-AKI-grade2plus': 'AKI',
    'target-death-in-30d': '30d death',
    'target-death-in-365d': '1y death',
    'target-ED-visit': 'ED visits',
    'target-esas-depression-3pt-change': 'Depression',
    'target-esas-pain-3pt-change': 'Pain',
    'target-esas-anxiety-3pt-change': 'Anxiety',
    'target-esas-tiredness-3pt-change': 'Tired',
    'target-esas-nausea-3pt-change': 'Nausea',
    'target-esas-drowsiness-3pt-change': 'Drowsy',
    'target-esas-appetite-3pt-change': 'Appetite',
    'target-esas-well-being-3pt-change': 'Well-being',
    'target-esas-shortness-of-breath-3pt-change': 'Dyspnea',
}

cancer_novelty_map = {
        # Seen (aerodigestive / thoracic / GI)
        "Bronchus and lung": 0,
        "Pancreas": 0,
        "Stomach": 0,
        "Colon": 0,
        "Rectum": 0,
        "Rectosigmoid junction": 0,
        "Small intestine": 0,
        "Esophagus": 0,
        "Liver and intrahepatic bile ducts": 0,
        "Gallbladder": 0,
        "Other and unspecified parts of biliary tract": 0,
        "Other and ill-defined digestive organs": 0,
        "Heart, mediastinum, and pleura": 0,
        "Thymus": 0,

        # Head & neck (seen)
        "Other and unspecified parts of mouth": 0,
        "Lip": 0,
        "Floor of mouth": 0,
        "Gum": 0,
        "Palate": 0,
        "Base of tongue": 0,
        "Other and unspecified parts of tongue": 0,
        "Tonsil": 0,
        "Oropharynx": 0,
        "Hypopharynx": 0,
        "Nasopharynx": 0,
        "Larynx": 0,
        "Pyriform sinus": 0,
        "Nasal cavity and middle ear": 0,
        "Accessory sinuses": 0,
        "Parotid gland": 0,
        "Other and unspecified major salivary glands": 0,
        "Other and ill-defined sites in lip, oral cavity, and pharynx": 0,

        # New
        "Skin": 1,
        "Breast": 1,
        "Bladder": 1,
        "Adrenal Gland": 1,
        "Prostate gland": 1,
        "Connective, subcataneous, and other soft tissues": 1,
        "Thyroid and other endocrine glands": 1,
        "Kidney": 1,
        "Renal pelvis": 1,
        "Ureter": 1,
        "Anus and anal canal": 1,
        "Brain": 1,
        "Peripheral nerves and autonomic nervous system": 1,
        "Spinal cord, cranial nerves, and other parts of central nervous system": 1,
        "Ovary": 1,
        "Corpus uteri": 1,
        "Cervix uteri": 1,
        "Uterus, NOS": 1,
        "Vulva": 1,
        "Penis": 1,
        "Testis": 1,
        "Placenta": 1,
        "Eye and adnexa": 1,
        "Bones, joints, and articular cartilage of limbs": 1,
        "Bones, joints, and articular cartilage of other and unspecified sites": 1,
        "Retroperitoneum and peritoneum": 1,
        "Other and unspecified urinary organs": 1,
        "Other and unspecified female genital organs": 1,
        "Unknown primary site": 1,
    }

_EPR_PHYSICIAN_ROWS = [
    ("Hamzeh AdelAmin Albaba", 2006, "N", "Arabic"),
    ("Raymond Jang", 2006, "Y", "NA"),
    ("Xueyu Eric Chen", 1996, "Y", "Mandarin"),
    ("Anna Spreafico", 2003, "N", "Italian"),
    ("Frances Alice Shepherd", 1970, "Y", "French"),
    ("Penelope Bradbury", 1994, "N", "NA"),
    ("David William Hedley", 1970, "N", "NA"),
    ("Natasha Basant Leighl", 1994, "Y", "NA"),
    ("Neesha Cindy Dhani", 2001, "Y", "NA"),
    ("Rebecca Michelle Prince", 2005, "N", "NA"),
    ("Grainne Mary O'Kane", 2007, "N", "NA"),
    ("Aaron Richard Hansen", 2004, "N", "NA"),
    ("Doreen Anuli Ezeife", 2011, "Y", "NA"),
    ("Geoffrey Liu", 1993, "Y", "NA"),
    ("Jennifer Jane Knox", 1995, "Y", "NA"),
    ("Yanshuo Cao", 2009, "N", "Mandarin"),
    ("Lawson Eng", 2014, "Y", "NA"),
    ("Adrian Gerold Sacher", 2009, "Y", "NA"),
    ("Aline Fusco Fares", 2010, "N", "Portuguese"),
    ("Kirstin Ann Perdrizet", 2013, "Y", "NA"),
    ("Shari Moura", 1991, "Y", "NA"),
    ("Mor Tal Moskovitz", 2010, "N", "Hebrew"),
    ("Sally Lau", 2013, "Y", "NA"),
    ("Michael Herman", 2013, "Y", "NA"),
    ("Di Maria Jiang", 2012, "Y", "Mandarin"),
    ("Lillian L. Y. Siu", 1991, "Y", "Cantonese"),
    ("Daniel Yokom", 2011, "Y", "NA"),
    ("Elaine Sarah Bouttell", 1997, "Y", "NA"),
    ("Ibrahim Algorashi", 2016, "N", "Arabic"),
    ("Kelvin Young", 2010, "N", "NA"),
    ("Daniel Shepshelovich", 2005, "N", "Hebrew, Russian"),
    ("Catherine Labbe", 2009, "Y", "French"),
    ("Kirsty Laura Taylor", 2011, "N", "NA"),
    ("Charles Lim", 2012, "Y", "NA"),
    ("Elena Elimova", 2008, "Y", "NA"),
    ("Hao-Wen Sim", 2007, "N", "NA"),
    ("Sara Soldera", 2011, "Y", "NA"),
    ("Ivan Lyra Gonzalez", 2011, "N", "Spanish"),
    ("Kyaw Aung", 2000, "N", "Burmese")
]

_EPIC_PHYSICIAN_ROWS = [
    ("Lawson Eng", 2014, "Y", "NA"),
    ("Anna Spreafico", 2003, "N", "Italian"),
    ("Raymond Jang", 2006, "Y", "NA"),
    ("Xueyu Eric Chen", 1996, "Y", "Mandarin"),
    ("Geoffrey Liu", 1993, "Y", "NA"),
    ("Di Maria Jiang", 2012, "Y", "Mandarin"),
    ("Penelope Bradbury", 1994, "N", "NA"),
    ("Lillian L. Y. Siu", 1991, "Y", "Cantonese"),
    ("Frances Alice Shepherd", 1970, "Y", "French"),
    ("Natasha Basant Leighl", 1994, "Y", "NA"),
    ("Enrique Sanz Garcia", 2010, "N", "Spanish"),
    ("Vikaash Kumar", 2005, "Y", "NA"),
    ("Erica Tsang", 2015, "Y", "NA"),
    ("Philippe Lucien Bedard", 2003, "Y", "French"),
    ("Abdulrahman Alghabban", 2013, "N", "Arabic"),
    ("Robert Charles Grant", 2014, "Y", "NA"),
    ("Cha Len Lee", 2009, "N", "Chinese"),
    ("Ma, Lucy Xiaolu", 2016, "Y", "NA"),
    ("Nazanin Fallah-Rad", 2010, "Y", "NA"),
    ("Richard Odwyer", 2014, "N", "NA"),
    ("Consolacion Molto Valiente", 2014, "N", "Spanish"),
    ("Lucy Corke", 2011, "N", "NA"),
    ("Jamie Feng", 2017, "Y", "NA"),
    ("Thais Megid", 2015, "N", "Spanish, Portuguese"),
    ("Sameena Khan", 2010, "N", "NA"),
    ("Eitan Amir", 2003, "N", "Hebrew"),
    ("Marie-Philippe Saltiel", 2016, "Y", "French"),
    ("Danielle Cuthbert", 2017, "N", "NA"),
    ("Massimo Di Iorio", 2018, "Y", "French, Italian"),
    ("Erika Martinez", 2013, "N", "Spanish"),
    ("Carly Barron", 2017, "Y", "NA"),
    ("Vikas Garg", 2011, "N", "Hindi"),
    ("Samuel David Saibil", 2010, "Y", "NA"),
    ("Ian Hirsch", 2017, "N", "French, Portuguese, Spanish"),
    ("Yacob Saleh", 2009, "N", "Arabic"),
    ("Abhenil Mittal", 2014, "N", "Hindi"),
    ("Sulaiman Almuthri", 2015, "N", "Arabic"),
    ("Gordon Moffat", 2016, "N", "NA"),
    ("Marcus Otho Butler", 1992, "N", "NA"),
    ("Abdul Rehman Farooq", 2009, "N", "Urdu"),
]

_COLUMNS = [
    "med_onc",
    "YOG",
    "Canadian_Medical_Graduate",
    "Other_Languages",
]

def _build_physician_char_df(physician_data) -> pd.DataFrame:
    df = pd.DataFrame(physician_data, columns=_COLUMNS)

    df = df.dropna(subset=["YOG"])

    df["Canadian_Medical_Graduate"] = df["Canadian_Medical_Graduate"] == "Y"
    df["Speaks_2nd_Language"] = df["Other_Languages"] != "NA"

    return df

df_physician_char_EPR = _build_physician_char_df(_EPR_PHYSICIAN_ROWS)
df_physician_char_EPIC = _build_physician_char_df(_EPIC_PHYSICIAN_ROWS)

# This dictionary maps detailed ICD-O-3 codes (C-codes) to a coarser, clinically relevant group.
CANCER_COARSE_SITE_MAP = {
    # Head & Neck (H&N)
    "C00": "Head & Neck", "C01": "Head & Neck", "C02": "Head & Neck", "C03": "Head & Neck", "C04": "Head & Neck", 
    "C05": "Head & Neck", "C06": "Head & Neck", "C07": "Head & Neck", "C08": "Head & Neck", "C09": "Head & Neck", 
    "C10": "Head & Neck", "C11": "Head & Neck", "C12": "Head & Neck", "C13": "Head & Neck", "C14": "Head & Neck", 
    "C30": "Head & Neck", "C31": "Head & Neck", "C32": "Head & Neck", 
    
    # Upper GI
    "C15": "Upper GI", "C16": "Upper GI", "C17": "Upper GI", 
    
    # Lower GI
    "C18": "Lower GI", "C19": "Lower GI", "C20": "Lower GI", "C21": "Lower GI", 
    
    # Hepato-Pancreato-Biliary (HPB)
    "C22": "HPB", "C23": "HPB", "C24": "HPB", "C25": "HPB", 
    
    # Thoracic (Lung & Chest)
    "C33": "Thoracic", "C34": "Thoracic", "C37": "Thoracic", "C38": "Thoracic", 
    
    # Musculoskeletal & Soft Tissue (Sarcomas)
    "C40": "Musculoskeletal & Soft Tissue", "C41": "Musculoskeletal & Soft Tissue", "C49": "Musculoskeletal & Soft Tissue",
    
    # Skin
    "C44": "Skin",
    
    # Breast
    "C50": "Breast",
    
    # Female Genital
    "C51": "Female Genital", "C52": "Female Genital", "C53": "Female Genital", "C54": "Female Genital", 
    "C55": "Female Genital", "C56": "Female Genital", "C57": "Female Genital", "C58": "Female Genital", 
    
    # Male Genital
    "C60": "Male Genital", "C61": "Male Genital", "C62": "Male Genital", "C63": "Male Genital", 
    
    # Urological
    "C64": "Urological", "C65": "Urological", "C66": "Urological", "C67": "Urological", "C68": "Urological", 
    
    # Nervous System
    "C69": "Nervous System", "C71": "Nervous System", "C72": "Nervous System", 
    
    # Endocrine
    "C73": "Endocrine", "C74": "Endocrine", "C75": "Endocrine", 
    
    # Other / Ill-Defined
    "C26": "Other / Ill-Defined", "C47": "Other / Ill-Defined", "C48": "Other / Ill-Defined", 
    "C76": "Other / Ill-Defined", "C80": "Other / Ill-Defined"
}

# Higher priority numbers mean the cancer site is considered more aggressive/prognostically significant 
# and should be chosen as the "Primary Site" in case of multiple primaries.
# This list is ordered from highest priority (index 0) to lowest priority.
CANCER_HIERARCHY = [
    "Thoracic",                         # 1 (Highest Priority)
    "HPB",                              # 2
    "Nervous System",                   # 3
    "Upper GI",                         # 4
    "Lower GI",                         # 5
    "Head & Neck",                      # 6
    "Female Genital",                   # 7
    "Urological",                       # 8
    "Male Genital",                     # 9
    "Breast",                           # 10
    "Endocrine",                        # 11
    "Musculoskeletal & Soft Tissue",    # 12
    "Skin",                             # 13
    "Other / Ill-Defined"               # 14 (Lowest Priority)
]
# Create a dictionary where the key is the coarse group and the value is its rank (lower rank means higher priority)
CANCER_HIERARCHY_RANK = {site: rank for rank, site in enumerate(CANCER_HIERARCHY)}