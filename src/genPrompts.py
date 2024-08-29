import json
import argparse


def genPersona():
    persona_prompt = [
        "Act as a highly experienced and extremely competent medical oncologist from the world-renowned Princess Margaret Cancer Centre in Toronto, Ontario. ",
        "Act as an expert medical oncologist. ",
        "Act as an incredibly skilled, well-trained machine learning model. ",
    ]

    return persona_prompt


def genDeid():
    deid_prompt = (
        "The following tags are used to replace the protected health information in the de-identified note.\n "
        + "PATIENT refers to the patient's name.\n "
        + "STAFF is an umbrella term for all hospital staff.\n "
        + "PHONE refers to phone number.\n "
        + "ID refers to any personal identification numbers.\n "
        + "EMAIL is any email address.\n "
        + "PATORG is any organization entity.\n "
        + "LOC refers to location.\n "
        + "HOSP refers to hospital related information.\n "
        + "OTHERPHI is a catch-all for other types of protected health information.\n"
    )

    return deid_prompt


def genTarget(target_string):
    additional_info = ""
    if target_string == "target_ED_visit":
        target_prompt = "visit the emergency department within the next 30 days"
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

        target_prompt = f"experience a {esas_change_value} point change in the ESAS score for {esas_target}"

        # extract the point change

        additional_info = (
            "The ESAS score refers to the Edmonton Symptom Assessment System. "
            + "It's a clinical tool used to assess the severity of common symptoms "
            + "experienced by patients with cancer and other advanced illnesses. Patients rate the "
            + "severity of each symptom on a scale from 0 to 10, with 0 indicating no symptom "
            + "and 10 indicating the worst possible severity. This assessment helps healthcare "
            + "providers manage symptoms and improve quality of life for patients. "
        )

    target_prompt = (
        "Your task is to predict the probability that a patient undergoing systemic therapy for cancer will "
        + target_prompt
        + " based on the de-identified clinical note below. "
    )
    target_prompt = target_prompt + additional_info

    return target_prompt


def genHealthFactors(target_string):
    health_string = ""
    if target_string == "target_death_in_365d":
        health_string = (
            "Consider all relevant factors, including the patient's current treatment regimen, "
            + "underlying cancer type and stage, symptoms, response to treatment, frequency of visits, "
            + "patterns in the Electronic Health Record, lifestyle factors, and any comorbid conditions. "
            + "Additionally, incorporate current medical guidelines and the latest research on survival rates and "
            + "risk factors associated with systemic cancer therapies. "
        )
    else:
        raise Exception("Not implemented yet.")

    return health_string


def genCOT():
    cot_prompt = (
        "Use chain of thought reasoning to logically deduce the patient's risk. "
        + "Clearly explain your reasoning step-by-step, referencing specific details from "
        + "the patient's EHR and how they contribute to the overall risk assessment. "
    )

    return cot_prompt


def genReasonExample(numeric_proba):
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

    return reason_prompt


def genProba(numeric_proba):
    if numeric_proba:
        # proba_prompt = 'Probability should be a value between 0 and 1 with 2 decimal digits. '
        proba_prompt = (
            "<PROBABILITY> should be a value between 0 and 1 with 2 decimal digits. "
        )
    else:
        # proba_prompt = "The response to 'Probability' must either be 'low', 'medium' or 'high'.  "
        proba_prompt = "<PROBABILITY> must either be 'low', 'medium' or 'high'.  "

    return proba_prompt


def genPrompts(target_name, numeric_proba, save_dir):
    persona_prompt = genPersona()
    target_prompt = genTarget(target_name)
    proba_prompt = genProba(numeric_proba)

    # format_prompt = "Provide a concise reasoning that explains how you arrived at the predicted probability. Express your response as a JSON object with the keys 'Reason' and 'Probability'.  "
    # format_prompt = "You will only respond with a JSON object with the keys 'Reason' and 'Probability'. The response to 'Reason' must be a concise explanation of how you arrived at the predicted probability.  "
    # format_prompt = ("""I am going to give you a template for your output. CAPITALIZED WORDS are my placeholders. Fill in my placeholders with your output. """ +
    #                 """Please preserve the overall formatting of my template. My template is:\n {'Reason': REASON, 'Probability': PROBABILITY}\n""" +
    #                 """REASON must be a very concise explanation of how you arrived at the predicted probability enclosed in double quotes. """)
    format_prompt = (
        """I am going to give you a template for your output. Words in angle brackets are my placeholders. Fill in my placeholders with your output. """
        + """Please preserve the overall formatting of my template. My template is:\n {'Reason': <REASON>, 'Probability': <PROBABILITY>}\n"""
        + """<REASON> must be a very concise explanation of how you arrived at the predicted probability enclosed in double quotes. """
    )
    format_end_prompt = "Ensure your output follows the above template strictly. "

    health_factors = genHealthFactors(target_name)
    health_prompt = ["", health_factors]

    cot_prompt = genCOT()
    reason_prompt = genReasonExample(numeric_proba)

    prompt_dict = {}

    ctr = 0
    for persona_val in persona_prompt:
        for health_val in health_prompt:
            for add_deid in [0, 1]:
                if add_deid == 1:
                    deid_prompt = genDeid()
                else:
                    deid_prompt = ""

                for cot_param in [0, 1]:
                    if cot_param == 1:
                        prompt = (
                            persona_val
                            + target_prompt
                            + health_val
                            + deid_prompt
                            + cot_prompt
                            + format_prompt
                            + proba_prompt
                            + format_end_prompt
                        )
                    else:
                        prompt = (
                            persona_val
                            + target_prompt
                            + health_val
                            + deid_prompt
                            + format_prompt
                            + proba_prompt
                            + format_end_prompt
                            + reason_prompt
                        )

                    prompt_dict[ctr] = prompt
                    ctr = ctr + 1

    # save dict to json
    fname = f"{save_dir}/promptList_{target_name}_numeric-proba{numeric_proba}.json"
    with open(fname, "w") as file:
        json.dump(prompt_dict, file, indent=4)

    # ask llm for comments on the prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_name", help="target name", type=str)  # name of target
    parser.add_argument(
        "numeric_proba", help="numeric value for probability", type=int
    )  # numeric value for probability?
    parser.add_argument("save_dir", help="save directory", type=str)  # save directory
    args = parser.parse_args()

    genPrompts(args.target_name, args.numeric_proba, args.save_dir)
