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
    database_path = './database/database.db'

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

    # 360split하는거
    split360_images(path_to_meshroom, save_360_frames_path, './', image_path)
    print("!!!done splitting 360 images!!!")
    
    
    #data split을 해줌
    total_image_num = 135 * 8
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
    
    
    feature_extractor(database_path, image_path)
    exhaustive_matcher(database_path)
    
    i = 1
    for (root2, dirs2, files2) in os.walk(image_list_path):
    #     # dirs1 = sorted(dirs1)
        files2 = sorted(files2)
    #     # print(dirs1)
        for file2 in files2:
            mkdir(f"{output_path}/{i}")
            mapper(database_path, image_path,f"{output_path}/{i}",os.path.join(root2, file2))
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
            if count_subdirectories(f"{output_path}/{i}") == 2:
                image_undistorter(image_path, f"{output_path}/{i}/1", f"{output_path}/{i}/undistorted")
            else:
                image_undistorter(image_path, f"{output_path}/{i}/0", f"{output_path}/{i}/undistorted")
            convert_colmap_bin_to_txt(f"{output_path}/{i}/undistorted/sparse", f"{output_path}/{i}/undistorted")
            export_model_to_ply(f"{output_path}/{i}/undistorted/sparse", f"{output_ply}/{i}.ply")
            
            i+=1

    #위에까지 colmap 돌리고 .ply파일 저장함
    submodel_num = i
    matrixs = []
    for j in range(submodel_num - 2):
        matrixs.append(np.array(get_merge_matrix(f"{output_path}/{j+1}/undistorted/images.txt",f"{output_path}/{j+2}/undistorted/images.txt")))

   
    for k in range(len(matrixs) - 1):
        matrixs[k+1] = np.dot(matrixs[k],matrixs[k+1])
    #이제 matrix들은 준비 된거임.
    

    for f in range(len(matrixs)):
        print(matrixs[f])
    print("-------------------------")



if __name__ == "__main__":
    main()