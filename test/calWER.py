import os

def calculate_wer(reference, hypothesis):
    # Calculate the WER of a single sentence, case-insensitive
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, reference.lower(), hypothesis.lower())
    matches = matcher.find_longest_match(0, len(reference), 0, len(hypothesis))
    return (len(reference) - matches.size) / len(reference) if reference else 0

def average_wer(ref_file, hyp_file):
    # Calculates the average WER, case-insensitive
    total_wer = 0.0
    ref_lines = 0
    with open(ref_file, 'r') as f_ref, open(hyp_file, 'r') as f_hyp:
        for ref_line, hyp_line in zip(f_ref, f_hyp):
            ref_text = ref_line.strip().lower().split()[1:]  # 去除行首的ID和空格，转换为小写
            hyp_text = hyp_line.strip().lower().split()[1:]
            total_wer += calculate_wer(' '.join(ref_text), ' '.join(hyp_text))
            ref_lines += 1

    # Check if the number of rows is inconsistent
    ref_file_len = sum(1 for _ in open(ref_file))
    hyp_file_len = sum(1 for _ in open(hyp_file))
    if ref_file_len != hyp_file_len:
        print(f"Warning: The number of lines in the reference file ({ref_file_len}) does not match the hypothesis file ({hyp_file_len}).")

    avg_wer = total_wer / ref_lines if ref_lines > 0 else 0
    return avg_wer

ref_file_path = '/root/ASR_repo/test_clean/text'
hyp_file_path = '/root/ASR_repo/result/1best_recog/text'

average_wer_value = average_wer(ref_file_path, hyp_file_path)
print(f"The average WER is: {average_wer_value * 100:.2f}%")