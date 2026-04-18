CTCAE_constants = {
    'hemoglobin': {
        'grade2plus': 100,
        'grade3plus': 80
    },
    'neutrophil': {
        'grade2plus': 1.5,
        'grade3plus': 1.0
    },
    'platelet': {
        'grade2plus': 75,
        'grade3plus': 50
    },
    'AKI': {
        'grade2plus': 1.5,
        'grade3plus': 3.0,
        'ULN': 98
    },
    'ALT': {
        'grade2plus': 3.0,
        'grade3plus': 5.0,
        'ULN': 40.0
    },
    'AST': {
        'grade2plus': 3.0,
        'grade3plus': 5.0,
        'ULN': 34.0
    },
    'bilirubin': {
        'grade2plus': 1.5,
        'grade3plus': 3.0,
        'ULN': 22.0
    }
}

map_CTCAE_lab = {'AKI': 'creatinine', 
                 'ALT': 'alanine_aminotransferase', 
                 'AST': 'aspartate_aminotransferase', 
                 'bilirubin': 'total_bilirubin'}

HEME_DEPTS = {
    "HEM", "LY", "CLL", "CML", "AML", "ALL", "APL",
    "MDS", "MPN", "WM", "MY",
    "BMT", "ALLO", "AUTO", "IEC"
}