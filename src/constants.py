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
    "C69": "Nervous System", "C70": "Nervous System", "C71": "Nervous System", "C72": "Nervous System", 
    
    # Endocrine
    "C73": "Endocrine", "C74": "Endocrine", "C75": "Endocrine", 
    
    # Other / Ill-Defined
    "C26": "Other / Ill-Defined", "C47": "Other / Ill-Defined", "C48": "Other / Ill-Defined", 
    "C76": "Other / Ill-Defined", "C80": "Other / Ill-Defined"
}

epr_map_cancer_to_group = {

    # Head & Neck (low-prevalence individually)
    "Nasopharynx": "Head and neck",
    "Tonsil": "Head and neck",
    "Larynx": "Head and neck",
    "Base of tongue": "Head and neck",
    "Other and unspecified parts of mouth": "Head and neck",
    "Other and unspecified parts of tongue": "Head and neck",
    "Oropharynx": "Head and neck",
    "Floor of mouth": "Head and neck",
    "Nasal cavity and middle ear": "Head and neck",
    "Pyriform sinus": "Head and neck",
    "Hypopharynx": "Head and neck",
    "Accessory sinuses": "Head and neck",
    "Parotid gland": "Head and neck",
    "Other and unspecified major salivary glands": "Head and neck",
    "Palate": "Head and neck",
    "Gum": "Head and neck",
    "Lip": "Head and neck",
    "Other and ill-defined sites in lip, oral cavity, and pharynx": "Head and neck",

    # GI (split high-prevalence vs rare)
    "Colon": "Colorectal",
    "Rectosigmoid junction": "Colorectal",
    "Rectum": "Colorectal",
    "Pancreas": "Pancreas",
    "Stomach": "Stomach",
    "Esophagus": "Esophagus",
    "Liver and intrahepatic bile ducts": "Liver",
    "Small intestine": "Other GI",
    "Gallbladder": "Other GI",
    "Other and unspecified parts of biliary tract": "Other GI",
    "Anus and anal canal": "Other GI",
    "Other and ill-defined digestive organs": "Other GI",

    # Thoracic
    "Bronchus and lung": "Lung",
    "Trachea": "Other thoracic",
    "Thymus": "Other thoracic",
    "Heart, mediastinum, and pleura": "Other thoracic",

    # GU (high-prevalence separate, rare grouped)
    "Prostate gland": "Prostate",
    "Bladder": "Bladder",
    "Kidney": "Kidney",
    "Testis": "Testis",
    "Penis": "Other GU",
    "Renal pelvis": "Other GU",
    "Ureter": "Other GU",
    "Other and unspecified urinary organs": "Other GU",
    "Other and unspecified male genital organs": "Other GU",

    # Gynecologic
    "Ovary": "Ovary",
    "Cervix uteri": "Cervix",
    "Corpus uteri": "Corpus uteri",
    "Uterus, NOS": "Corpus uteri",
    "Vulva": "Other gyn",
    "Vagina": "Other gyn",
    "Other and unspecified female genital organs": "Other gyn",
    "Placenta": "Other gyn",

    # Breast
    "Breast": "Breast",

    # CNS
    "Brain": "Brain/CNS",
    "Spinal cord, cranial nerves, and other parts of central nervous system": "Brain/CNS",
    "Eye and adnexa": "Brain/CNS",

    # Endocrine
    "Thyroid and other endocrine glands": "Thyroid",
    "Adrenal Gland": "Other endocrine",
    "Other endocrine glands and related structures": "Other endocrine",

    # Sarcoma (usually rare, grouped)
    "Bones, joints, and articular cartilage of limbs": "Sarcoma",
    "Bones, joints, and articular cartilage of other and unspecified sites": "Sarcoma",
    "Connective, subcataneous, and other soft tissues": "Sarcoma",
    "Retroperitoneum and peritoneum": "Sarcoma",

    # Skin
    "Skin": "Skin",

    # Ill-defined
    "Other and ill-defined sites": "Other/Ill-defined",
}

def epic_map_cancer_to_group(raw_cancer_type):
    name = raw_cancer_type.lower()
    # Substring-based automatic mapping
    if "pancreas" in name:
        return "Pancreas"
    if "bladder" in name:
        return "Bladder"
    if "liver" in name or "hepatic" in name:
        return "Liver"
    if "kidney" in name or "renal" in name:
        return "Kidney"
    if "breast" in name or "areola" in name or "nipple" in name:
        return "Breast"
    if "stomach" in name or "gastric" in name or "pylorus" in name:
        return "Stomach"
    if "testis" in name or "testes" in name:
        return "Testis"
    if "ovary" in name or "tubo-ovarian" in name:
        return "Ovary"
    if "thyroid" in name:
        return "Thyroid"
    if "prostate" in name:
        return "Prostate"
    if "esophagus" in name or "gastroesophageal" in name or "cardia" in name:
        return "Esophagus"
    if "lung" in name or "bronchus" in name:
        return "Lung"
    if "skin" in name:
        return "Skin"
    if "pleura" in name or "thymus" in name or "mediastinum" in name or "heart" in name:
        return "Other thoracic"
    if "corpus uteri" in name or "endometrium" in name or "myometrium" in name:
        return "Corpus uteri"
    if "placenta" in name:
        return "Other gyn"
    if "vulva" in name:
        return "Other gyn"
    if "vagina" in name:
        return "Other gyn"
    if "cervix" in name:
        return "Cervix"
    if "ureter" in name:
        return "Other GU"
    if "penis" in name or "spermatic cord" in name:
        return "Other GU"
    if "adrenal" in name:
        return "Other endocrine"
    if "parotid" in name or "salivary gland" in name or "nasal" in name or "oral" in name or "oropharynx" in name or "larynx" in name or "tonsil" in name or "tongue" in name or "pharynx" in name or "gum" in name or "lip" in name or "cheek" in name or "mouth" in name or "palate" in name or "maxillary" in name or "mandible" in name or "sinus" in name or "temple" in name or "ear" in name or "eye" in name or "orbit" in name or "scalp" in name:
        return "Head and neck"
    if "brain" in name or "spinal" in name or "meninges" in name or "cerebral" in name or "lobe" in name or "cns" in name or "thalamus" in name or "optic chiasm" in name or "cerebellum" in name:
        return "Brain/CNS"
    if "appendix" in name or "colon" in name or "cecum" in name or "sigmoid" in name or "rectum" in name or "rectosigmoid" in name:
        return "Colorectal"
    if "anus" in name or "anal" in name or "anorectum" in name:
        return "Other GI"
    if "gallbladder" in name or "biliary" in name or "ampulla of vater" in name:
        return "Other GI"
    if "retroperitoneum" in name or "peritoneum" in name or "omentum" in name:
        return "Sarcoma"
    if "connective" in name or "sarcoma" in name or "soft tissues" in name or "bone" in name or "femur" in name or "sacrum" in name or "ilium" in name:
        return "Sarcoma"
    if "small intestine" in name:
        return "Other GI"
    if "ill-defined digestive" in name or "digestive organs" in name:
        return "Other GI"
    if "peripheral nerve" in name or "autonomic nervous system" in name:
        return "Sarcoma"
    if "ill-defined" in name or "unknown" in name or "overlapping" in name:
        return "Other/Ill-defined"
    if "female genital organs" in name:
        return "Other gyn"
    if "urinary" in name:
        return "Other GU"
    # Hardcoded exceptions for ambiguous or rare cases
    hardcoded = {
        "Areola": "Breast",
        "Fallopian tube": "Other gyn",
        "Nipple": "Breast",
        "Lacrimal gland": "Head and neck",
        "Ciliary body": "Brain/CNS",
        "Undescended testis (site of neoplasm)": "Testis",
        "Spermatic cord": "Other GU",
        "Unknown primary site": "Other/Ill-defined",
        "Uterus, NOS": "Corpus uteri",   # generic
        "Endometrium": "Corpus uteri",
        "Myometrium": "Corpus uteri",
        "Soft tissues, NOS": "Sarcoma",
        "Gastrointestinal tract, NOS": "Other GI",
        "Nasal cavity": "Head and neck",
        "Maxillary sinus": "Head and neck",
        "Oropharynx, NOS": "Head and neck",
        "Tonsil, NOS (excludes Lingual tonsil C02.4 and Pharyngeal tonsil C11.1)": "Head and neck",
        "Buccal mucosa": "Head and neck",
        "Hypopharyngeal wall": "Head and neck",
        "Glottis": "Head and neck",
        "Sole of foot": "Skin",
        "Female genital organs, NOS": "Other gyn",
        "Cardioesophageal junction": "Esophagus",
        "Corpus callosum": "Brain/CNS",
        "Gastrointestinal tract, NOS": "Other GI",
        "Duodenum": "Other GI",  # small bowel
        "Urinary organs, overlapping lesion": "Other GU",
        "Epiglottis, anterior surface": "Head and neck",
        "Choroid": "Brain/CNS",  # grouped with eye/CNS
        "Subglottis": "Head and neck",
        "Submandibular gland": "Head and neck",
        "Supraglottis": "Head and neck",
        "Alveolus, NOS": "Head and neck",
        "Concha": "Head and neck",
        "Urinary system, NOS": "Other GU",
        "Ileum": "Other GI",
        "Epiglottis, NOS": "Head and neck",
        "Uveal tract": "Brain/CNS",  # grouped with eye/CNS
        "Uterus, NOS": "Corpus uteri",
        "Conjunctiva": "Brain/CNS",  # grouped with eye/CNS
        "Nervous system, NOS": "Brain/CNS",
        "Small bowel, NOS": "Other GI",
        "Unknown primary site": "Other/Ill-defined",
        "Peripheral nerves/ANS of abdomen": "Sarcoma",
        "Alveolus, lower": "Head and neck",
        "Bile duct, NOS": "Other GI",
        "Pancreatic duct": "Pancreas",
        "Basal ganglia": "Brain/CNS",
        "Alveolar mucosa, lower": "Head and neck",
        "Common bile duct": "Other GI",
        "Gingiva, mandibular": "Head and neck",
        "Uterine, NOS": "Corpus uteri",
        "Pinna": "Head and neck",
        "Retromolar trigone": "Head and neck",
        "Vocal cord, NOS": "Head and neck",
        "Auricle, NOS": "Head and neck",
        "Ala nasi": "Head and neck"
        # Add more as you encounter other special cases
    }
    return hardcoded.get(raw_cancer_type, "Other/Ill-defined")

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

# cancer site / cancer morphology
CANCER_CODE_MAP = {
    "C00": "Lip",
    "C01": "Base of tongue",
    "C02": "Other and unspecified parts of tongue",
    "C03": "Gum",
    "C04": "Floor of mouth",
    "C05": "Palate",
    "C06": "Other and unspecified parts of mouth",
    "C07": "Parotid gland",
    "C08": "Other and unspecified major salivary glands",
    "C09": "Tonsil",
    "C10": "Oropharynx",
    "C11": "Nasopharynx",
    "C12": "Pyriform sinus",
    "C13": "Hypopharynx",
    "C14": "Other and ill-defined sites in lip, oral cavity, and pharynx",
    "C15": "Esophagus",
    "C16": "Stomach",
    "C17": "Small intestine",
    "C18": "Colon",
    "C19": "Rectosigmoid junction",
    "C20": "Rectum",
    "C21": "Anus and anal canal",
    "C22": "Liver and intrahepatic bile ducts",
    "C23": "Gallbladder",
    "C24": "Other and unspecified parts of biliary tract",
    "C25": "Pancreas",
    "C26": "Other and ill-defined digestive organs",
    "C30": "Nasal cavity and middle ear",
    "C31": "Accessory sinuses",
    "C32": "Larynx",
    "C33": "Trachea",
    "C34": "Bronchus and lung",
    "C37": "Thymus",
    "C38": "Heart, mediastinum, and pleura",
    "C40": "Bones, joints, and articular cartilage of limbs",
    "C41": "Bones, joints, and articular cartilage of other and unspecified sites",
    "C43": "Skin (melanoma)",
    "C44": "Skin",
    "C47": "Peripheral nerves and autonomic nervous system",
    "C48": "Retroperitoneum and peritoneum",
    "C49": "Connective, subcataneous, and other soft tissues",
    "C50": "Breast",
    "C51": "Vulva",
    "C52": "Vagina",
    "C53": "Cervix uteri",
    "C54": "Corpus uteri",
    "C55": "Uterus, NOS",
    "C56": "Ovary",
    "C57": "Other and unspecified female genital organs",
    "C58": "Placenta",
    "C60": "Penis",
    "C61": "Prostate gland",
    "C62": "Testis",
    "C63": "Other and unspecified male genital organs",
    "C64": "Kidney",
    "C65": "Renal pelvis",
    "C66": "Ureter",
    "C67": "Bladder",
    "C68": "Other and unspecified urinary organs",
    "C69": "Eye and adnexa",
    "C70": "Meninges",
    "C71": "Brain",
    "C72": "Spinal cord, cranial nerves, and other parts of central nervous system",
    "C73": "Thyroid and other endocrine glands",
    "C74": "Adrenal Gland",
    "C75": "Other endocrine glands and related structures",
    "C76": "Other and ill-defined sites",
    "C77": "Lymph nodes",
    "C80": "Unknown primary site",
    "800": "Neoplasms, NOS",
    "801": "Epithelial neoplasms, NOS",
    "802": "Epithelial neoplasms, NOS",
    "803": "Epithelial neoplasms, NOS",
    "804": "Epithelial neoplasms, NOS",
    "805": "Squamous cell neoplasms",
    "807": "Squamous cell neoplasms",
    "808": "Squamous cell neoplasms",
    "809": "Basal cell neoplasms",
    "812": "Transitional cell papillomas and carcinomas",
    "813": "Transitional cell papillomas and carcinomas",
    "814": "Adenomas and adenocarcinomas",
    "816": "Adenomas and adenocarcinomas",
    "817": "Adenomas and adenocarcinomas",
    "818": "Adenomas and adenocarcinomas",
    "820": "Adenomas and adenocarcinomas",
    "821": "Adenomas and adenocarcinomas",
    "823": "Adenomas and adenocarcinomas",
    "824": "Adenomas and adenocarcinomas",
    "825": "Adenomas and adenocarcinomas",
    "826": "Adenomas and adenocarcinomas",
    "829": "Adenomas and adenocarcinomas",
    "831": "Adenomas and adenocarcinomas",
    "832": "Adenomas and adenocarcinomas",
    "833": "Adenomas and adenocarcinomas",
    "834": "Adenomas and adenocarcinomas",
    "837": "Adenomas and adenocarcinomas",
    "838": "Adenomas and adenocarcinomas",
    "840": "Adnexal and skin appendage neoplasms",
    "843": "Mucoepidermoid neoplasms",
    "844": "Cystic, mucinous, and serous neoplasms",
    "845": "Cystic, mucinous, and serous neoplasms",
    "846": "Cystic, mucinous, and serous neoplasms",
    "847": "Cystic, mucinous, and serous neoplasms",
    "848": "Cystic, mucinous, and serous neoplasms",
    "849": "Cystic, mucinous, and serous neoplasms",
    "850": "Ductal and lobular neoplasms",
    "851": "Ductal and lobular neoplasms",
    "852": "Ductal and lobular neoplasms",
    "853": "Ductal and lobular neoplasms",
    "854": "Ductal and lobular neoplasms",
    "855": "Acinar cell neoplasms",
    "856": "Complex epithelial neoplasms",
    "857": "Complex epithelial neoplasms",
    "858": "Thymic epithelial neoplasms",
    "862": "Specialized gonadal neoplasms",
    "872": "Paragangliomas and glomus tumors",
    "873": "Nevi and melanomas",
    "874": "Nevi and melanomas",
    "877": "Nevi and melanomas",
    "880": "Soft tissue tumors and sarcomas, NOS",
    "881": "Fibromatous neoplasms",
    "883": "Fibromatous neoplasms",
    "885": "Lipomatous neoplasms",
    "889": "Myomatous neoplasms",
    "890": "Myomatous neoplasms",
    "891": "Myomatous neoplasms",
    "892": "Myomatous neoplasms",
    "893": "Complex mixed and stromal neoplasms",
    "894": "Complex mixed and stromal neoplasms",
    "895": "Complex mixed and stromal neoplasms",
    "898": "Complex mixed and stromal neoplasms",
    "902": "Fibroepithelial neoplasms",
    "904": "Synovial-like neoplasms",
    "905": "Mesothelial neoplasms",
    "906": "Germ cell neoplasms",
    "907": "Germ cell neoplasms",
    "908": "Germ cell neoplasms",
    "910": "Trophoblastic neoplasms",
    "911": "Mesonephromas",
    "912": "Blood vessel tumors",
    "914": "Blood vessel tumors",
    "915": "Blood vessel tumors",
    "918": "Osseous and chondromatous neoplasms",
    "922": "Osseous and chondromatous neoplasms",
    "924": "Osseous and chondromatous neoplasms",
    "926": "Miscellaneous bone tumors",
    "936": "Miscellaneous tumors",
    "938": "Gliomas",
    "940": "Gliomas",
    "942": "Gliomas",
    "944": "Gliomas",
    "945": "Gliomas",
    "947": "Gliomas",
    "954": "Nerve sheath tumors",
    "956": "Nerve sheath tumors",
}

# DRUG constants

epr_CHEMO_DRUGS = [
    'GEMCITABINE HCL',
    'NAB-PACLITAXEL(ABRAXANE)',
    'OXALIPLATIN',
    'CISPLATIN',
    'IRINOTECAN HCL',
    'ETOPOSIDE',
    'FLUOROURACIL',
    'LEUCOVORIN CALCIUM',  # folinic acid, not cytotoxic itself but part of chemo regimens
    'CARBOPLATIN',
    'PACLITAXEL',
    'MITOMYCIN',
    'VINORELBINE TARTRATE',
    'DOCETAXEL',
    'EPIRUBICIN HCL',
    'PEMETREXED DISODIUM',
    'GEMCITABINE - PAID',
    'LIPOSOMAL IRINOTECAN (ONIVYDE)',
    'CYCLOPHOSPHAMIDE',
    'DOXORUBICIN HCL',
    'VINCRISTINE SULFATE',
    'RALTITREXED',
    'CAPECITABINE TRIAL SUPPLY',  # oral fluoropyrimidine
    'FLUOROURACIL STUDY SUPPLY',
    'CISPLATIN STUDY SUPPLY',
    'DOCETAXEL - PAID',
    'DOXORUBICIN STUDY SUPPLY'
]

epr_CHEMO_DRUG_MAP = {
    'GEMCITABINE HCL': 'GEMCITABINE',
    'GEMCITABINE TRIAL SUPPLY': 'GEMCITABINE',
    'GEMCITABINE - PAID': 'GEMCITABINE',
    
    'NAB-PACLITAXEL(ABRAXANE)': 'NAB-PACLITAXEL',
    'NAB-PACL(ABRAXANE) STUDY SUPPL': 'NAB-PACLITAXEL',
    
    'OXALIPLATIN': 'OXALIPLATIN',
    'OXALIPLATIN TRIAL SUPPLY': 'OXALIPLATIN',
    
    'CISPLATIN': 'CISPLATIN',
    'CISPLATIN STUDY SUPPLY': 'CISPLATIN',
    
    'IRINOTECAN HCL': 'IRINOTECAN',
    'LIPOSOMAL IRINOTECAN (ONIVYDE)': 'IRINOTECAN (LIPOSOMAL)',
    
    'ETOPOSIDE': 'ETOPOSIDE',
    
    'FLUOROURACIL': 'FLUOROURACIL',
    'FLUOROURACIL STUDY SUPPLY': 'FLUOROURACIL',
    
    'LEUCOVORIN CALCIUM': 'LEUCOVORIN',
    
    'CARBOPLATIN': 'CARBOPLATIN',
    'CARBOPLATIN STUDY SUPPLY': 'CARBOPLATIN',
    
    'PACLITAXEL': 'PACLITAXEL',
    'PACLITAXEL STUDY SUPPLY': 'PACLITAXEL',
    
    'MITOMYCIN': 'MITOMYCIN',
    
    'VINORELBINE TARTRATE': 'VINORELBINE',
    
    'DOCETAXEL': 'DOCETAXEL',
    'DOCETAXEL - PAID': 'DOCETAXEL',
    
    'EPIRUBICIN HCL': 'EPIRUBICIN',
    
    'PEMETREXED DISODIUM': 'PEMETREXED',
    'PEMETREXED STUDY SUPPLY': 'PEMETREXED',
    
    'CYCLOPHOSPHAMIDE': 'CYCLOPHOSPHAMIDE',
    
    'DOXORUBICIN HCL': 'DOXORUBICIN',
    'DOXORUBICIN STUDY SUPPLY': 'DOXORUBICIN',
    
    'VINCRISTINE SULFATE': 'VINCRISTINE',
    
    'RALTITREXED': 'RALTITREXED',

    'CAPECITABINE TRIAL SUPPLY': 'CAPECITABINE'
}

epic_CHEMO_DRUG_MAP = {
    # --- Antimetabolites ---
    "fluorouracil": "FLUOROURACIL",
    "capecitabine": "CAPECITABINE",
    "gemcitabine": "GEMCITABINE",
    "methotrexate": "METHOTREXATE",
    "cytarabine": "CYTARABINE",
    "azacitidine": "AZACITIDINE",
    "fludarabine": "FLUDARABINE",
    "cladribine": "CLADRIBINE",
    "nelarabine": "NELARABINE",
    "mercaptopurine": "MERCAPTOPURINE",
    "pralatrexate": "PRALATREXATE",

    # --- Platinums ---
    "cisplatin": "CISPLATIN",
    "carboplatin": "CARBOPLATIN",
    "oxaliplatin": "OXALIPLATIN",

    # --- Topoisomerase inhibitors ---
    "irinotecan": "IRINOTECAN",
    "irinotecan hydrochloride trihydrate": "IRINOTECAN",
    "etoposide": "ETOPOSIDE",
    "topotecan": "TOPOTECAN",

    # --- Alkylating agents ---
    "cyclophosphamide": "CYCLOPHOSPHAMIDE",
    "ifosfamide": "IFOSFAMIDE",
    "melphalan": "MELPHALAN",
    "busulfan": "BUSULFAN",
    "dacarbazine": "DACARBAZINE",
    "temozolomide": "TEMOZOLOMIDE",
    "lomustine": "LOMUSTINE",
    "carmustine": "CARMUSTINE",
    "bendamustine": "BENDAMUSTINE",
    "chlorambucil": "CHLORAMBUCIL",
    "streptozocin": "STREPTOZOCIN",
    "thiotepa": "THIOTEPA",
    "treosulfan": "TREOSULFAN",
    "procarbazine": "PROCARBAZINE",

    # --- Anthracyclines / related ---
    "doxorubicin": "DOXORUBICIN",
    "pegylated_liposomal_doxorubicin": "DOXORUBICIN",
    "epirubicin": "EPIRUBICIN",
    "daunorubicin": "DAUNORUBICIN",
    "idarubicin": "IDARUBICIN",
    "mitoxantrone": "MITOXANTRONE",

    # --- Microtubule agents ---
    "paclitaxel": "PACLITAXEL",
    "docetaxel": "DOCETAXEL",
    "cabazitaxel": "CABAZITAXEL",
    "vincristine": "VINCRISTINE",
    "vinorelbine": "VINORELBINE",
    "vinblastine": "VINBLASTINE",
    "eribulin": "ERIBULIN",

    # --- Other cytotoxics ---
    "bleomycin": "BLEOMYCIN",
    "mitomycin": "MITOMYCIN",
    "dactinomycin": "DACTINOMYCIN",
    "amsacrine": "AMSACRINE",
    "lurbinectedin": "LURBINECTEDIN",
    "trabectedin": "TRABECTEDIN",
    "asparaginase": "ASPARAGINASE",
    "pegaspargase": "ASPARAGINASE",
    "arsenic trioxide": "ARSENIC TRIOXIDE",
    "hydroxyurea": "HYDROXYUREA",

    # --- Folate modulation (kept since paired with chemo) ---
    "leucovorin": "LEUCOVORIN",
    "folinic acid": "LEUCOVORIN",
    "calcium folinate": "LEUCOVORIN",
    "raltitrexed": "RALTITREXED",
}