import os
import shutil

def copy_files_containing_string(src_dir, dest_dir, string):
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                contents = f.read()
            if string in contents:
                shutil.copy(file_path, dest_dir)


src_dir = "/home/suresh/Desktop/Night_Vision_Rendering/evaluation/MBNet_result_images"  #source directory
dest_dir = "/home/suresh/Desktop/Night_Vision_Rendering/evaluation/MBNet_selected_images"  #destination directory
string = '_boost_HIS_03'  #the string you want to search for

copy_files_containing_string(src_dir, dest_dir, string)
print("program executed successfully!")
