"""
	Data Tools
	BioCapsule
	Edwin Sanchez
	
	Description:
	These are tools for working with the MOBIO dataset.
	- integrity check
	- file extraction
"""

# importing the "tarfile" module
import tarfile
import os
import hashlib


def main():
	extract()


# check the integrity of the data using the hash file for the compressed files	
def check_integrity():
	# input file path
	hash_file_path = "./MD5SUM.TXT"
	
	# open the file
	with open(hash_file_path, "r") as file:
		for line in file:
			line = line.strip()
			
			# find where to split the line
			index = find_spaces(line)
			
			# get the split info
			text_hash, file_name = split_text(line, index)

			# compute MD5SUM on file_name
			# BUF_SIZE is totally arbitrary, change for your app!
			BUF_SIZE = 524_288_000  # lets read stuff in 64kb chunks!

			md5 = hashlib.md5()
			with open(f"./{file_name}", 'rb') as f:
				while True:
					data = f.read(BUF_SIZE)
					if not data:
						break
					md5.update(data)
			
			# check if it matches the given hash
			current_hash = md5.hexdigest()
			is_same = (current_hash == text_hash)
			
			# print info
			print(file_name)
			print(f"Current Hash:  {current_hash}")
			print(f"Original Hash: {text_hash}")
			print(is_same)
			
		
# find where the spaces are that separate the hash from the file name
def find_spaces(line:str)->int:
	for i, char in enumerate(line):
		if char == " ":
			return i
		
# split the text based on the first space index
def split_text(line:str, index:int)-> tuple:
	text_hash = line[:index]
	file_name = line[index+2:]
	return (text_hash, file_name)
	

# extract the contents of the files in a directory
def extract():
    # get all files in dir
    all_files = os.listdir("./")

    # loop over all files
    extract_file_count = 0
    for file_name in all_files:
        # check if the file is a .tar.gz file
        if file_name.endswith(".tar.gz"):
            extract_file_count += 1
            print(f"Extracting {file_name}...")
            # open file
            with tarfile.open(file_name) as file:
                # get base name of compressed file
                base_name = os.path.splitext(os.path.splitext(os.path.basename(file_name))[0])[0]

                print(base_name)

                path_to_extract_to = f"./{base_name}/"
                
                # create the dir if it doesn't exist
                if os.path.exists(path_to_extract_to) == False:
                    os.mkdir(path_to_extract_to)
                else:
                    with open("./already_exists.txt", "a") as app_file:
                        print(f"{base_name} already exists!")
                        app_file.write(base_name)

                # extract file
                file.extractall(path_to_extract_to)
    print(f"Num Extracted files: {extract_file_count}")


if __name__ == "__main__":
    main()

