# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', default='datasets/lvis/lvis_v1_val.json')
    args = parser.parse_args()

    print('Loading', args.ann)
    data = json.load(open(args.ann, 'r'))
    catid2freq = {x['id']: x['frequency'] for x in data['categories']}
    
    img_id_list = {}
    image_new = []
    for a in data["annotations"]:
        if catid2freq[a['category_id']] == "r":
            if img_id_list.get(a['image_id'], None) == None:   
                img_id_list[a['image_id']] = True
            
    for i in data["images"]:
        if img_id_list.get(i['id'], False) == True:
            image_new.append(i)
    print(len(image_new))
    
    
    print('ori #anns', len(data['annotations']))
    exclude = ['f', 'c']
    data['annotations'] = [x for x in data['annotations'] \
        if catid2freq[x['category_id']] not in exclude]
    print('filtered #anns', len(data['annotations']))
    
    data["images"] = image_new
    
    out_path = args.ann[:-5] + '_rare.json'
    print('Saving to', out_path)
    json.dump(data, open(out_path, 'w'))
