def target_category(target: str) -> str:
    """
    Categorize target based on naming patterns.
    
    Parameters:
    -----------
    target : str
        Target name
        
    Returns:
    --------
    str
        Target category ('lab', 'symptom', or 'clinic')
    """
    if 'grade2plus' in target:
        return 'lab'
    elif 'change' in target:
        return 'symptom'
    else:
        return 'clinic'