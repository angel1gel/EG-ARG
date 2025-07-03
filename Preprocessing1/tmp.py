import os

# 配置路径
pdb_list_file = "/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing1/input.txt"  # 替换为你的txt文件路径
label_folder = "/ifs/home/dongyihan/protein/EquiPNAS/Protein_train_data/label"       # 替换为你的label文件夹路径
output_file = "/ifs/home/dongyihan/protein/EquiPNAS/Preprocessing1/output.txt"          # 结果保存路径
# 结果保存路径

# 读取PDB文件名列表
with open(pdb_list_file, "r") as file:
    pdb_files = [line.strip() for line in file]

# 存储结果
results = []

for pdb in pdb_files:
    label_file_path = os.path.join(label_folder, f"{pdb.split('.')[0]}.label")  # 假设标签文件以 .label 结尾

    if os.path.exists(label_file_path):
        # 读取标签文件
        with open(label_file_path, "r") as label_file:
            label_data = label_file.read().strip()  # 读取整个字符串并去除空白字符
            
        count_label_1 = label_data.count("1")  # 统计 '1' 的数量

        # 保存结果
        results.append((pdb, count_label_1))
    else:
        print(f"标签文件未找到: {label_file_path}")

# 按标签为1的数量排序
results.sort(key=lambda x: x[1], reverse=True)

# 保存到文件
with open(output_file, "w") as out_file:
    for pdb, count in results:
        out_file.write(f"{pdb}: {count}\n")

print(f"结果已保存到 {output_file}")
