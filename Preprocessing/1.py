# import os

# # 指定文件夹路径
# folder_path = "/ifs/home/dongyihan/Data/pdb-variant"
# output_file = "/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing/input1/target.txt"

# # 获取文件夹中所有文件
# file_names = os.listdir(folder_path)

# # 筛选出 .pdb 文件，并获取文件名（不包括扩展名）
# pdb_names = [os.path.splitext(file)[0] for file in file_names if file.endswith('.pdb')]

# # 将结果写入到 txt 文件中
# with open(output_file, 'w') as f:
#     for name in pdb_names:
#         f.write(name + '\n')

# print(f"所有的 .pdb 文件名已写入到 {output_file} 中。")

# import os
# import shutil

# # 源文件夹路径
# source_folder = "/ifs/home/dongyihan/protein/GraphBind/Datasets/PBBB/feature/Trans"
# # 目标文件夹路径
# destination_folder = "/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing/input1"

# # 确保目标文件夹存在，不存在则创建
# if not os.path.exists(destination_folder):
#     os.makedirs(destination_folder)

# # 遍历源文件夹中的所有文件
# for file_name in os.listdir(source_folder):
#     # 构造完整文件路径
#     source_path = os.path.join(source_folder, file_name)
#     destination_path = os.path.join(destination_folder, file_name)

#     # 确保只复制文件（忽略子文件夹）
#     if os.path.isfile(source_path):
#         shutil.copy(source_path, destination_path)
#         print(f"已复制: {file_name}")

# print("文件复制完成！")

# import os

# # 文件夹路径
# folder_path = "/ifs/home/dongyihan/protein/GraphBind/Datasets/PBBB/feature/Trans"

# # 遍历文件夹中的所有文件
# for file_name in os.listdir(folder_path):
#     # 构造文件的完整路径
#     file_path = os.path.join(folder_path, file_name)

#     # 检查文件是否是 .fasta 文件
#     if os.path.isfile(file_path) and file_name.endswith('.fasta'):
#         # 删除文件
#         os.remove(file_path)
#         print(f"已删除: {file_name}")

# print("删除完成！")


# import os

# # 输入的多序列 .fasta 文件路径
# input_fasta = "/ifs/home/dongyihan/Data/variant.fasta"
# # 输出文件夹路径
# output_folder = "/ifs/home/dongyihan/Data/fasta"

# # 确保输出文件夹存在，不存在则创建
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # 打开并读取输入 .fasta 文件
# with open(input_fasta, 'r') as infile:
#     content = infile.read()

# # 按序列分割，确保每个序列以 '>' 开头
# sequences = content.strip().split('>')
# sequences = [seq for seq in sequences if seq]  # 去掉空字符串

# # 遍历每个序列并写入单独的 .fasta 文件
# for seq in sequences:
#     lines = seq.splitlines()
#     header = lines[0]  # 获取序列头
#     sequence = "\n".join(lines[1:])  # 获取序列内容

#     # 生成输出文件路径，以序列头命名
#     output_file = os.path.join(output_folder, f"{header.split()[0]}.fasta")

#     # 将序列写入到单独的 .fasta 文件中
#     with open(output_file, 'w') as outfile:
#         outfile.write(f">{header}\n{sequence}\n")

#     print(f"已写入: {output_file}")

# print("拆分完成！")


# import os

# # 文件夹路径
# folder_path = "/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing/input1"

# # 获取文件夹中所有文件和子文件夹
# all_items = os.listdir(folder_path)

# # 筛选出文件（排除子文件夹）
# files = [item for item in all_items if os.path.isfile(os.path.join(folder_path, item))]

# # 统计文件数量
# file_count = len(files)

# print(f"文件夹中共有 {file_count} 个文件。")

# import os

# # 文件夹路径
# pdb_folder = "/ifs/home/dongyihan/protein/GraphBind/Datasets/PBBB/PDB"
# fasta_folder = "/ifs/home/dongyihan/Data/fasta"

# # 获取 .pdb 文件夹中的文件名（去掉后缀）
# pdb_files = [os.path.splitext(file)[0] for file in os.listdir(pdb_folder) if file.endswith('.pdb')]

# # 获取 .fasta 文件夹中的文件名（去掉后缀）
# fasta_files = [file for file in os.listdir(fasta_folder) if file.endswith('.fasta')]

# # 遍历 .fasta 文件夹中的文件
# for fasta_file in fasta_files:
#     fasta_name = os.path.splitext(fasta_file)[0]
    
#     # 如果 .fasta 文件名不在 .pdb 文件名列表中，删除该文件
#     if fasta_name not in pdb_files:
#         file_path = os.path.join(fasta_folder, fasta_file)
#         os.remove(file_path)
#         print(f"已删除: {file_path}")

# print("删除完成！")

# # 输入 TXT 文件路径
# input_txt = "/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing/input1/target.txt"
# # 输出 TXT 文件路径
# output_txt = "/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing/input1/target1.txt"

# # 打开输入文件并读取所有行
# with open(input_txt, 'r') as infile:
#     lines = infile.readlines()

# # 在每行末尾添加 .pdb
# updated_lines = [line.strip() + '.pdb\n' for line in lines]

# # 写入到输出文件
# with open(output_txt, 'w') as outfile:
#     outfile.writelines(updated_lines)

# print(f"已处理完成，结果保存到 {output_txt}")


# import os
# import shutil

# import os
# import shutil

# # 原始文件夹路径（混乱的文件夹）
# source_folder = "/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing/temp"
# # 目标文件夹路径（用于存放 .dist 文件）
# destination_folder = "/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing/distmaps1"

# # 确保目标文件夹存在，不存在则创建
# if not os.path.exists(destination_folder):
#     os.makedirs(destination_folder)

# # 遍历源文件夹中的所有文件
# for file_name in os.listdir(source_folder):
#     # 构造完整文件路径
#     source_path = os.path.join(source_folder, file_name)

#     # 检查是否为 .dist 文件
#     if os.path.isfile(source_path) and file_name.endswith('.dist'):
#         # 构造目标文件路径
#         destination_path = os.path.join(destination_folder, file_name)
#         # 复制文件
#         shutil.copy(source_path, destination_path)
#         print(f"已复制: {file_name}")

# print("提取完成！")


# import os

# # 文件夹路径
# dist_folder = "/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing/distmaps1"
# fasta_folder = "/ifs/home/dongyihan/Data/fasta"

# # 遍历 .dist 文件夹中的所有文件
# for dist_file in os.listdir(dist_folder):
#     if dist_file.endswith('.dist'):
#         # 构造对应的 fasta 文件名（假设去除后缀匹配）
#         fasta_file = dist_file.replace('.dist', '.fasta')
#         fasta_path = os.path.join(fasta_folder, fasta_file)

#         # 检查对应的 fasta 文件是否存在
#         if os.path.exists(fasta_path):
#             # 读取 fasta 文件中的序列（假设只有一个序列）
#             with open(fasta_path, 'r') as fasta:
#                 lines = fasta.readlines()
#                 sequence = ''.join(lines[1:]).strip()  # 提取序列内容，去除首尾空格

#             # 打开 dist 文件并将序列写入第一行
#             dist_path = os.path.join(dist_folder, dist_file)
#             with open(dist_path, 'r+') as dist:
#                 dist_lines = dist.readlines()
#                 dist_lines.insert(0, f"{sequence}\n")  # 在第一行插入序列

#                 # 覆盖写回文件
#                 dist.seek(0)
#                 dist.writelines(dist_lines)
#             print(f"已更新: {dist_file}")
#         else:
#             print(f"未找到对应的 .fasta 文件: {fasta_file}")

# print("处理完成！")


import os

# 文件夹路径
folder_path = "/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing1/esm"

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    # 构造文件的完整路径
    file_path = os.path.join(folder_path, file_name)

    # 检查文件是否是 .npy 文件
    if os.path.isfile(file_path) and file_name.endswith('.npy'):
        # 构造新的文件名
        new_file_name = file_name.replace('.npy', '_esm.npy')
        
        # 构造新的文件路径
        new_file_path = os.path.join(folder_path, new_file_name)
        
        # 重命名文件
        os.rename(file_path, new_file_path)
        print(f"已重命名: {file_name} -> {new_file_name}")

print("重命名完成！")
