input_filename = '/root/ASR_repo/test_clean/wav.scp'
output_filename = '/root/ASR_repo/test_clean/wav.scp_modified'

# 打开输入文件和输出文件
with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
    # 逐行读取文件
    for line in infile:
        # 替换空格为制表符
        line.split()
        modified_line = line.replace(' ', '\t')
        # 写入到输出文件
        outfile.write(modified_line)

print(f"File '{input_filename}' has been modified and saved as '{output_filename}'.")