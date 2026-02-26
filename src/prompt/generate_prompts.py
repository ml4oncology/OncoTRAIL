import json
import re
import argparse
from oncotrail.prep.constants import (
    CTCAE_constants)
import logging
logger = logging.getLogger(__name__)


def generate_persona():
    persona_prompt = [
        ("Act as a highly experienced and extremely competent medical oncologist " + 
         "from the world-renowned Princess Margaret Cancer Centre in Toronto, Ontario. "),
        "Assume you are a medical oncologist. ",
        "Assume you are an advanced medical AI model. ",
    ]

    return persona_prompt

def generate_deid():
    deid_prompt = (
        "The following tags are used to replace the protected health information in the de-identified note.\n"
        + "PATIENT refers to the patient's name.\n"
        + "STAFF is an umbrella term for all hospital staff.\n"
        + "PHONE refers to phone number.\n"
        + "ID refers to any personal identification numbers.\n"
        + "EMAIL is any email address.\n"
        + "PATORG is any organization entity.\n"
        + "LOC refers to location.\n"
        + "HOSP refers to hospital related information.\n"
        + "OTHERPHI is a catch-all for other types of protected health information.\n"
    )

    return deid_prompt

def generate_target_description(target_string, 
                                simplify, 
                                repeated_sampling,
                                include_task_instruction,
                                use_question_mark):

    logger.info(f"target_string: {target_string}")

    time_period = "within the next 30 days"
    additional_info = ""
    if target_string == "target_ED_visit":
        target_prompt = f"visit the emergency department {time_period}"
    elif target_string == "target_death_in_365d":
        target_prompt = "die within the next year"
    elif target_string == "target_death_in_30d":
        target_prompt = "die within the next 30 days"
    elif "esas" in target_string:
        if "well_being" in target_string:
            target_string = target_string.replace("well_being", "well-being")

        if "shortness_of_breath" in target_string:
            target_string = target_string.replace(
                "shortness_of_breath", "shortness of breath"
            )

        # extract the target
        esas_target = target_string.split("_")[2]
        esas_change_value = target_string.split("_")[3][0]

        if simplify == 0:
            target_prompt = (f"experience a {esas_change_value} point change in the ESAS score for {esas_target}" +
                             ' ' + time_period)

            # extract the point change

            additional_info = (
                "The ESAS score refers to the Edmonton Symptom Assessment System. "
                + "It's a clinical tool used to assess the severity of common symptoms "
                + "experienced by patients with cancer and other advanced illnesses. Patients rate the "
                + "severity of each symptom on a scale from 0 to 10, with 0 indicating no symptom "
                + "and 10 indicating the worst possible severity. This assessment helps healthcare "
                + "providers manage symptoms and improve quality of life for patients. "
            )
        else:
            target_prompt = f"experience worsening {esas_target} {time_period}"

    elif re.search(r'grade\d+plus', target_string) is not None:

        match = re.search(r'target_(.*?)_grade([1-5])plus', target_string)
        # extract the grade
        grade = match.group(2)
        # extract the quantity
        quantity = match.group(1)

        CTCAE_meaning = "defined by the CTCAE (Common Terminology Criteria for Adverse Events)"

        if "hemoglobin" in target_string:
            if simplify == 0:
                target_prompt = (
                    f"experience grade {grade} and above anemia {time_period}, {CTCAE_meaning} "
                    f"as a hemoglobin level under {CTCAE_constants[quantity][f'grade{grade}plus']} g/L"
                )
            else:
                target_prompt = f"experience worsening anemia {time_period}"

        elif "neutrophil" in target_string:
            if simplify == 0:
                target_prompt = (
                    f"experience grade {grade} and above neutrophil count decrease {time_period}, {CTCAE_meaning} "
                    f"as a neutrophil count under {CTCAE_constants[quantity][f'grade{grade}plus']} x 10e9/L"
                )
            else:
                target_prompt = f"experience worsening neutrophil count {time_period}"

        elif "platelet" in target_string:
            if simplify == 0:
                target_prompt = (
                    f"experience grade {grade} and above platelet count decrease {time_period}, {CTCAE_meaning} "
                    f"as a platelet count under {CTCAE_constants[quantity][f'grade{grade}plus']} x 10e9/L"
                )
            else:
                target_prompt = f"experience worsening platelet count {time_period}"

        elif "AKI" in target_string:
            if simplify == 0:
                target_prompt = (
                    f"experience grade {grade} and above creatinine increase {time_period}, {CTCAE_meaning} "
                    f"as creatinine increasing {CTCAE_constants[quantity][f'grade{grade}plus']} times above "
                    f"baseline or {CTCAE_constants[quantity][f'grade{grade}plus']} times above " 
                    f"the upper limit of normal ({CTCAE_constants[quantity]['ULN']} umol/L)"
                )
            else:
                target_prompt = f"experience acute kidney injury {time_period}"

        elif "ALT" in target_string or "AST" in target_string:
            if 'ALT' in target_string:
                quantity_full = 'alanine aminotransferase'

            elif 'AST' in target_string:
                quantity_full = 'aspartate aminotransferase'

            if simplify == 0:
                target_prompt = (
                    f"experience grade {grade} and above {quantity_full} increase {time_period}, {CTCAE_meaning} "
                    f"as {quantity_full} increasing {CTCAE_constants[quantity][f'grade{grade}plus']} times above "
                    f"the upper limit of normal ({CTCAE_constants[quantity]['ULN']} U/L) or baseline if the baseline was abnormal"
                )
            else:
                target_prompt = f"experience increasing {quantity_full} level {time_period}"

        elif "bilirubin" in target_string:
            if simplify == 0:
                target_prompt = (
                    f"experience grade {grade} and above blood bilirubin increase {time_period}, {CTCAE_meaning} "
                    f"as blood bilirubin increasing {CTCAE_constants[quantity][f'grade{grade}plus']} times above "
                    f"the upper limit of normal ({CTCAE_constants[quantity]['ULN']} umol/L) or baseline if the baseline was abnormal"
                )
            else:
                target_prompt = f"experience increasing blood bilirubin level {time_period}"

    # Add task instruction if requested
    if include_task_instruction:
        prefix = "Your task is to predict the probability that the patient will " if repeated_sampling == 0 \
            else "Your task is to predict whether the patient will "
        target_prompt = prefix + target_prompt

    # Add punctuation
    target_prompt += "?" if use_question_mark else "."

    # Add additional info if requested
    target_prompt += " " + additional_info

    return target_prompt

def generate_health_factors(target_string):

    health_string = (
        "Consider all relevant factors, including the patient's current treatment regimen, "
        + "underlying cancer type and stage, symptoms, response to treatment, frequency of visits, "
        + "patterns in the Electronic Health Record, lifestyle factors, and any comorbid conditions. "
        + "Additionally, incorporate current medical guidelines, the latest research on survival rates, "
        + "risk factors associated with systemic cancer therapies, and the latest medical research in general. "
    )

    return health_string

def generate_cot():
    cot_prompt = (
        "Use chain of thought reasoning to logically deduce the patient's risk. "
        + "Clearly explain your reasoning step-by-step, referencing specific details from "
        + "the patient's EHR and how they contribute to the overall risk assessment. "
    )

    return cot_prompt

def generate_reason_example(numeric_proba, target_string):
    if target_string == "target_death_in_365d":
        if numeric_proba == 1:
            prob_val = 0.95
        else:
            prob_val = "'high'"
        reason_prompt = "Below is an example output:\n"
        reason_prompt = reason_prompt + (
            """{'Reason': "Based on the patient's advanced age (70), extensive liver metastases, and rapid deterioration, """
            + """it is highly likely that the patient's condition will not improve with systemic therapy. The patient's inability """
            + """to undergo chemotherapy as an outpatient and the fact that the medical oncologist has already advised against chemotherapy """
            + """due to the patient's poor performance status suggest that the patient's prognosis is poor. The patient's symptoms, """
            + """including weight loss, abdominal pain, and bedridden status, also indicate a high likelihood of mortality within the next year.", """
            + f"""'Probability': {prob_val}"""
            + """}"""
        )
    else:
        reason_prompt = ''

    return reason_prompt

def generate_proba(numeric_proba):
    if numeric_proba:
        proba_prompt = (
            "Probability should be a value between 0 and 1. "
        )
    else:
        proba_prompt = "Probability must either be 'low', 'medium' or 'high'. "

    return proba_prompt

def generate_prompts(target_names, numeric_proba, save_dir, repeated_sampling, clinical_bench=0):

    # 0->23: non-simplified
    # 0->7: persona 1
    # 0->3: no health factors 
    # 0->1: no deid tags
    # 0: no cot
    # 1: with cot
    # 2->3: with deid tags
    # 4->7: with health factors
    # 8->15: persona 2
    # 16->23: persona 3

    # 24->47: simplified
    # 24->31: persona 1 
    # 32->39: persona 2
    # 40->47: persona 3

    list_of_targets = target_names.split(",")

    persona_prompt = generate_persona()
    proba_prompt = generate_proba(numeric_proba)

    if repeated_sampling == 0:
        format_prompt = ("You will only respond with a JSON object with the keys Reason and Probability. " +
                        "Reason must be a very concise explanation of how you arrived at the predicted probability. " +
                        proba_prompt +
                        """Example output: {"Reason": "<Your Reason>", "Probability": 0.5}.""")
    else:
        format_prompt = ("You will only respond with a JSON object with the keys Reason and Prediction. " +
                        "Reason must be a very concise explanation of how you arrived at your prediction. " +
                        "Prediction must be either 0 or 1. " +
                        """Example output: {"Reason": "<Your Reason>", "Prediction": 1}.""")
    if clinical_bench:
        note_details_prompt = ("You are reviewing a de-identified, text-formatted summary of clinical data from the past 30 days " + 
                           "for a patient receiving systemic cancer therapy on <TREATMENT DATE>. ")
    else:
        note_details_prompt = ("You are reviewing a de-identified clinical note below combining the initial consult and a recent note from the past 30 days " + 
                           "for a patient receiving systemic cancer therapy on <TREATMENT DATE>. ")

    for target_name in list_of_targets:
        health_factors = generate_health_factors(target_name)
        health_prompt = ["", health_factors]

        cot_prompt = generate_cot()
        reason_prompt = generate_reason_example(numeric_proba, target_name)

        prompt_dict = {}

        ctr = 0 
        for simplify in [0, 1]:
            target_prompt = generate_target_description(target_name, 
                                                        simplify, 
                                                        repeated_sampling,
                                                        True, False)
            for persona_val in persona_prompt:
                for health_val in health_prompt:
                    for add_deid in [0, 1]:
                        if add_deid == 1:
                            deid_prompt = generate_deid()
                        else:
                            deid_prompt = ""

                        for cot_param in [0, 1]:
                            if cot_param == 1:
                                prompt = (
                                    persona_val
                                    + note_details_prompt
                                    + target_prompt
                                    + health_val
                                    + deid_prompt
                                    + cot_prompt
                                    + format_prompt
                                )
                            else:
                                prompt = (
                                    persona_val
                                    + note_details_prompt
                                    + target_prompt
                                    + health_val
                                    + deid_prompt
                                    + format_prompt
                                    # + reason_prompt
                                )

                            prompt_dict[ctr] = prompt
                            ctr = ctr + 1

        # save dict to json
        fname = f"{save_dir}/prompt_list_{target_name}_numeric-proba{numeric_proba}.json"
        with open(fname, "w") as file:
            json.dump(prompt_dict, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('target_names', type=str, help='Comma-separated list of targets') # targets
    parser.add_argument(
        "numeric_proba", help="numeric value for probability", type=int
    )  # numeric value for probability?
    parser.add_argument("save_dir", help="save directory", type=str)  # save directory
    parser.add_argument("repeated_sampling", help="repeated sampling", type=int) # repeated sampling
    parser.add_argument("--clinical_bench", help = "use clinical bench?", type = int, default=0) # use clinical bench?
    args = parser.parse_args()
    generate_prompts(args.target_names, args.numeric_proba, args.save_dir, args.repeated_sampling, args.clinical_bench)
