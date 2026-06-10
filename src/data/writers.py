
def write_file_one_column_batch(batched_entry, column_name, row_ids=None):
    if row_ids is not None and len(row_ids) != len(batched_entry):
         raise ValueError("The number of entries within batch entry should be the exact same as columns")
    
    if row_ids is None:
        for data in batched_entry:
            record = { #
                    "prompt": "",
                    "concept": "",
                    "question_type": "",
                    "student_response": "",
                    "extracted_entities": [],
                    "entity_context": [],
                    "entity_labels": [],
                    "entity_confidence_labels": [],
                    "entity_verification_notes": [],
                    "intervention": [],
                    "intervention_label": [],
                    "retraction_reward": [],
                    "retraction_reward_notes": [],
                    "correction_reward": [],
                    "correct_reward_notes": [],
                } 
            record[column_name] = data
        #append bulk append
    else:
        for data,row_id in zip(batched_entry, row_ids):
            record = get_data(row_id)
            record[column_name] = data
            #in place_write

             
            
for example, output in zip(batch_examples, outputs):
        for completion in output.outputs:
            total_output_tokens += len(completion.token_ids)

           "prompt": output.prompt,
                "concept": example.get("topic"),
                "question_type": None,
                "student_response": completion.text,
                "extracted_entities": [],
                "entity_context": [],
                "entity_labels": [],
                "entity_confidence_labels": [],
                "entity_verification_notes": [],
                "intervention": [],
                "intervention_label": [],
                "retraction_reward": [],
                "retraction_reward_notes": [],
                "correction_reward": [],
                "correct_reward_notes": [], 

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            batch_records += 1

    f.flush()