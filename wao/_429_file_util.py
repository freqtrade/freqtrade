import os

from wao.config import Config


def delete_429_file(text):
    print("delete_429_file:...")
    execution_id = text.split('\n')[3].split(':')[1].replace(" ", "").replace("*", "")
    action_1 = text.split('\n')[0].split(':')[1].replace(" ", "").replace("*", "")
    action_2 = text.split('\n')[0].split(':')[2].replace(" ", "").replace("*", "")
    file_name = Config._429_DIRECTORY + execution_id + '_' + action_1 + '_' + action_2
    if os.path.isfile(file_name):
        os.remove(file_name)


def write_to_429_file(text):
    execution_id = text.split('\n')[3].split(':')[1].replace(" ", "").replace("*", "")
    action_1 = text.split('\n')[0].split(':')[1].replace(" ", "").replace("*", "")
    action_2 = text.split('\n')[0].split(':')[2].replace(" ", "").replace("*", "")
    with open(Config._429_DIRECTORY + execution_id + '_' + action_1 + '_' + action_2,
              'w') as file:
        file.write(text)
    file.close()
    print(
        "write_to_429_file: " + Config._429_DIRECTORY + execution_id + '_' + action_1 + '_' + action_2)
