import sys
from tools.vid_to_img import *
from tools.colmap_cmd import *
import os
from tools.get_merging_matrix import *
from tools.split360 import *
import numpy as np

def count_subdirectories(path):
    """주어진 경로에 있는 하위 디렉토리 수를 반환합니다."""
    # 주어진 경로에 있는 항목들의 리스트를 가져옵니다.
    entries = os.listdir(path)
    
    # entries 중에서 디렉토리인 것들만 필터링합니다.
    subdirectory_count = sum(os.path.isdir(os.path.join(path, entry)) for entry in entries)
    
    return subdirectory_count

def main():
    #colmap에서 만들어지는 database pwd
    database_path = 'database.db'

    #8방위로 찢은 이미지 dir
    image_path = './8_dir_frames'

    #colmap 과정 저장 dir
    output_path = './colmap_output'

    output_ply = './ply_files'

    image_list_path = './image_list'

    save_360_frames_path = './360frames'
    
    path_to_meshroom = '/home/mrlab/Meshroom-2023.3.0/aliceVision/bin/./aliceVision_split360Images'
    
    image_num = vid_to_img(sys.argv[1],save_360_frames_path)
    
    print("!!!done splitting frames!!!")

    # # 360split하는거
    split360_images(path_to_meshroom, save_360_frames_path, './', image_path)
    print("!!!done splitting 360 images!!!")
    
    
    #data split을 해줌
    ##############327 image_num으로 무조건 바꿔야함!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    total_image_num = image_num * 8
    submodel_image_num = 320
    merge_num = 80
    tmp = submodel_image_num
    submodel_num = 1
    while(tmp < total_image_num):
        tmp += (submodel_image_num - merge_num)
        submodel_num += 1
    print(submodel_num)
    # 왜 submodel_num -1을 넣냐면 마지막 꼬랑지가 scale이 안맞아서 merge가 이쁘게 안됨 그래서 걍 버림
    create_overlapping_image_lists(image_path,submodel_num - 1,submodel_image_num,merge_num)
    print("!!!done creating submodel text files!!!")
    
    
    
    def extract_number(filename):
        # 파일 이름에서 숫자 부분만 추출
        number = ''.join([char for char in filename if char.isdigit()])
        return int(number) if number else 0

    
    i = 1
    for (root2, dirs2, files2) in os.walk(image_list_path):
        files2 = sorted(files2, key=extract_number)
        for file2 in files2:
            mkdir(f"{output_path}/{i}")
            print(f"makeing progress with {i}th submodel...")
            feature_extractor(f"{output_path}/{i}/{database_path}", image_path, os.path.join(root2, file2))
            print(f"makeing progress with {i}th submodel...")
            exhaustive_matcher(f"{output_path}/{i}/{database_path}")
            mapper(f"{output_path}/{i}/{database_path}", image_path,f"{output_path}/{i}",os.path.join(root2, file2))
            print("***************")
            print("***************")
            print("***************")
            print("***************")
            print("***************")
            print(f"model {i} done")
            print("***************")
            print("***************")
            print("***************")
            print("***************")
            print("***************")
            a = count_subdirectories(f"{output_path}/{i}")
            if a >= 2:
                image_undistorter(image_path, f"{output_path}/{i}/{a-1}", f"{output_path}/{i}/undistorted",os.path.join(root2, file2))
            else:
                image_undistorter(image_path, f"{output_path}/{i}/0", f"{output_path}/{i}/undistorted", os.path.join(root2, file2))
            convert_colmap_bin_to_txt(f"{output_path}/{i}/undistorted/sparse/0", f"{output_path}/{i}/undistorted")
            export_model_to_ply(f"{output_path}/{i}/undistorted/sparse/0", f"{output_ply}/{i}.ply")
            
            i+=1

    #위에까지 colmap 돌리고 .ply파일 저장함
    submodel_num = i
    matrixs = []
    for j in range(submodel_num - 2):
        matrixs.append(np.array(get_merge_matrix(f"{output_path}/{j+1}/undistorted/images.txt",f"{output_path}/{j+2}/undistorted/images.txt")))

   
    for k in range(len(matrixs) - 1):
        matrixs[k+1] = np.dot(matrixs[k],matrixs[k+1])
    #이제 matrix들은 준비 된거임.
    
    for i in range(len(matrixs)):
        path = f'./matrix/{i+2}.txt'
        with open(path, 'w') as file:
            for row in matrixs[i]:
                for num in row:
                    file.write(str(num) + '\t')
                file.write('\n')
                
            

    for f in range(len(matrixs)):
        print(matrixs[f])
    print("-------------------------")



if __name__ == "__main__":
    main()