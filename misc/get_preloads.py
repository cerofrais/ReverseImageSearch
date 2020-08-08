#from misc import *
# from misc_imports import *
from os.path import join as pjoin
from os import environ
from azure.storage.blob import BlockBlobService


BLOB_ACCOUNT_NAME = environ['BLOB_ACCOUNT_NAME']
BLOB_ACCOUNT_PASS = environ['BLOB_ACCOUNT_PASS']

block_blob_service = BlockBlobService(
    account_name=BLOB_ACCOUNT_NAME,
    account_key=BLOB_ACCOUNT_PASS)

preloads_path = 'data/preloads' # when seen from root path.

print('downloading preloads from blob')

for blob in block_blob_service.list_blobs('contractspreloads'):
    #print(blob.name)
    block_blob_service.get_blob_to_path(
        'contractspreloads', blob.name, pjoin(preloads_path, blob.name))
