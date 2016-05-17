#!/bin/env python
# -*- coding: utf-8 -*-
import glob
import json
import urllib
import requests
import shutil


BASE_URL = 'http://media.style.com/image/ts'

import os
import errno

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def main():

    base_data_path = ['data/release_runway_0.1/data/collections/', '../data/release_runway_0.1/data/collections/']

    files = glob.glob(base_data_path[0] + '*.json')
    if not len(files):
        files = glob.glob(base_data_path[1] + '*.json')
        base_data_path = base_data_path[1]
    else:
        base_data_path = base_data_path[0]

    for filepath in files:
        collection_name = os.path.basename(filepath).split('.')[0]
        images_path = base_data_path + collection_name
        make_sure_path_exists(images_path)

        with open(filepath) as json_file:
            collection = json.load(json_file)

        for look in collection['looks']:
            slide_id = look['slideId']
            counter = 0

            for image in look['runwaySlide']['images']:
                image_url = BASE_URL + image['url']
                image_dest = images_path + '/' + slide_id + '_' + str(counter)

                r = requests.get(image_url, stream=True)
                if r.status_code == 200:
                    with open(image_dest, 'wb') as f:
                        r.raw.decode_content = True
                        shutil.copyfileobj(r.raw, f)

                    counter += 1
        try:
            os.rmdir(images_path)
            print('No image downloaded from collection {0}!'.format(collection_name))
        except OSError as ex:
            # move on, some image was downloaded
            pass

if __name__ == '__main__':
    main()