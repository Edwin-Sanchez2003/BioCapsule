"""
    Puts each participant's files
    into their own directory

    Easier for us later on
"""


import os
import shutil


MOBIO_LOCATIONS = [
    "but/",
    "idiap/",
    "lia/",
    "uman/",
    "unis/",
    "uoulu/"
] # end mobio locations


def main():
    base_dir = "../../../../../../media/edwsanch/One Touch/MOBIO_extracted/one_sec_intervals/"
    input_dirs = []
    input_dirs.append(os.path.join(base_dir, MOBIO_LOCATIONS[0]))
    input_dirs.append(os.path.join(base_dir, MOBIO_LOCATIONS[2]))

    # loop over the given directories
    for input_dir in input_dirs:
        # get all of the p_names
        p_names = get_all_ids(input_dir=input_dir)

        # all file_names
        file_names = os.listdir(input_dir)

        # move p_name files to their own dir
        for p_name in p_names:
            move_to_own_dir(base_dir=input_dir,
                            file_names=file_names,
                            participant_id=p_name)


# gets all of the ids for a particular input directory
def get_all_ids(input_dir:str)-> "list[str]":
    file_names = os.listdir(input_dir)
    p_names = []
    for f_name in file_names:
        p_name = f_name[13:17]
        if is_in_list(p_name, p_names) == False:
            p_names.append(p_name)
    return p_names
        

def is_in_list(item, arr:list)-> bool:
    for val in arr:
        if val == item:
            return True
    return False


# moves a participant's files to their own directory
def move_to_own_dir(base_dir:str, file_names:str, participant_id:str)-> None:
    # make the dir if it doesn't already exist
    new_loc = os.path.join(base_dir, participant_id)
    if os.path.isdir(new_loc) == False:
        os.makedirs(new_loc)
    
    # search through file names, graph relevant ones
    for f_name in file_names:
        # grab spot that has the names of participants
        p_name = f_name[13:17]
        
        # check if belongs to this participant
        if p_name == participant_id:
            # put file into new dir
            f_path = os.path.join(base_dir, f_name)
            if os.path.isdir(f_path) == False:
                shutil.move(src=f_path, dst=new_loc)
            else:
                print(f_path)



if __name__ == "__main__":
    main()
