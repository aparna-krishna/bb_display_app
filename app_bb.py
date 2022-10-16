import streamlit as st
import pandas as pd
import os
import torch
import fastai
from PIL import Image

import urllib.request

  

def get_rec_outfit(test_df,img_index, image_files, embeddings, occasion="", query_cat="", num_options=1):
    

    cat_map = {'womens-dresses':'dress',
                'womens-jumpsuits-and-rompers':'dress',
                'womens-shirt':'tops',
               'womens-cardigans':'tops',
               'womens-tees':'tops',
               'womens-blouse':'tops',
               'womens-bodysuit':'tops',
               'womens-bralette':'tops',
               'womens-playsuit':'dress',
               'womens-singlets':'tops',
               'womens-jumpers':'tops',
               'womens-regular-pants':'bottoms',
               'womens-coats-and-jackets':'tops',
               'sneakers':'shoes',
               'boots':'shoes',
               'sandals':'sandals',
               'bag':'bag',
               'belt':'none',
               'Belt':'none',
               'Socks':'none',
               'socks':'none',
               'womens-pants':'bottoms',
               'womens-jeans':'bottoms',
               'womens-shorts':'bottoms',
               'womens-skirts':'bottoms',
               'womens-joggers':'bottoms',
               'womens-leggings':'bottoms'}

    current_img = embeddings[img_index]
    
    cosine = torch.nn.CosineSimilarity(dim=0)
    similarity_dict={}
    for idx, embedding in enumerate(embeddings):
        similarity_dict[idx]=cosine(current_img, embedding)
    
    x = {k: v for k, v in sorted(similarity_dict.items(), key=lambda item: item[1])}
        

    query_img_pth =f'../data/retailer_test_sets/bluebungalow_2k_single_image/{image_files[img_index]}'
    #st.write(query_img_pth)

    cat = str(test_df[test_df['ImgPath']==query_img_pth]['category'].tolist()[0].lower())
    cat = cat_map[cat]

    outfit_dict_types = {'dress':['bag', 'sandals'],
    'top':['bottom', 'shoes'],
    'bottom':['top','shoes'],
    'shoes':['top', 'bottom']}

    outfit = {
        'query':query_img_pth, 
        "item1":{"cat":outfit_dict_types[query_cat][0], "suggestion":[]},
        "item2":{"cat":outfit_dict_types[query_cat][1],"suggestion":[]}
        }


    df_path_prefix = '../data/retailer_test_sets/bluebungalow_2k_single_image'

    
    for im_idx,val in x.items():
        # if yet to generate correct number suggestion for either of the items, then perform logic
        if len(outfit["item1"]["suggestion"]) < num_options or len(outfit['item2']['suggestion']) < num_options:
            curr_img_pth = f'{df_path_prefix}/{(image_files[im_idx])}'
            curr_cat = str(test_df[test_df['ImgPath']==curr_img_pth]['category'].tolist()[0].lower())
            curr_cat = cat_map[curr_cat]
            curr_occ="" # TODO: ignore occasion for current retailer prototype
            if occasion=="":
                if outfit['item1']['cat'] in curr_cat:
                    if len(outfit['item1']['suggestion']) < num_options:
                        outfit['item1']['suggestion'].append(curr_img_pth)
                if outfit['item2']['cat'] in curr_cat:
                    if len(outfit['item2']['suggestion']) < num_options:
                        outfit['item2']['suggestion'].append(curr_img_pth)
            else:
                if outfit['item1']['cat'] in curr_cat and occasion in curr_occ:
                    if len(outfit['item1']['suggestion']) < num_options:
                        outfit['item1']['suggestion']=curr_img_pth
                if outfit['item2']['cat'] in curr_cat and occasion in curr_occ:
                    if len(outfit['item1']['suggestion']) < num_options:
                        outfit['item1']['suggestion'].append(curr_img_pth)
                
        else:
            return outfit
    
    

# load the test set + embeddings
#test_df = pd.read_csv('bb_results_df.csv') 
test_df = pd.read_csv('bb_cleaner.csv') 
ims = test_df.ImgPath.tolist()#os.listdir('bluebungalow_2k_single_image')
image_files=[]
for im in ims: image_files.append(im.split('/')[-1])
embeddings=torch.load('bb_test_set_embeddings.pt')

# create the sidebar
with st.sidebar:
    idx = st.number_input(label='Query Idx',min_value=0, max_value=1999)
    cat=st.radio('Query Item:',['dress', 'top', 'bottom', 'shoes'])

    occ=""#st.radio('Occasion:', ['Casual', 'Formal', 'Sports'])

    st.session_state['query_category'] = cat
    st.session_state['query_occasion'] = occ
    st.session_state['query_idx'] = int(idx)

    st.write('Test Set stats:')
    
    st.write(test_df.occasion.value_counts())
    st.write(test_df.category.value_counts())





# create the outfit columns
#col1,col2,col3 = st.columns([1,1,1], gap="small")
col1,col2 = st.columns(2, gap="small")

# get IDX from query image path
# cat of query image ( translate okkular category to general cat )
translate={'top':'womens-blouse', 'bottom':'womens-regular-pants', 'dress':'womens-dresses', 'shoes':'sandals'}
query_img = test_df[test_df['category']==translate[st.session_state['query_category']]]['ImgPath'].tolist()[int(st.session_state['query_idx'])]
folder_format = query_img.split('/')[-1]
idx=image_files.index(folder_format)

# get the recommended outfit
outfit = get_rec_outfit(test_df, idx, image_files, embeddings, st.session_state['query_occasion'], st.session_state['query_category'], num_options=4)

st.session_state['outfit'] = outfit
if st.session_state['query_category'] == 'bottom':
    st.session_state['outfit_order'] = ['top', 'shoes']
elif st.session_state['query_category'] == 'top':
    st.session_state['outfit_order'] = ['bottom', 'shoes']
elif st.session_state['query_category'] == 'shoes':
    st.session_state['outfit_order'] = ['bottom', 'top']
elif st.session_state['query_category'] == 'dress':
    st.session_state['outfit_order'] = ['sandals', 'bag']
elif st.session_state['query_category'] == 'shoes':
    st.session_state['outfit_order'] = ['top', 'bottom']



def get_concat_v(im1, im2, query):


    urllib.request.urlretrieve(
    im1,
    "im1.jpg")
    urllib.request.urlretrieve(
    im2,
    "im2.jpg")
  
    im1 = Image.open("im1.jpg")
    im2 = Image.open("im2.jpg")


    #if isinstance(query,str):query=Image.open(query)
    if isinstance(im1,str):im1=Image.open(im1)
    if isinstance(im2,str):im2=Image.open(im2)
    #offset=200
    offset=0
    #im1 = im1.resize((int(query.height*0.8), int(query.width)), Image.ANTIALIAS)
    #im2 = im2.resize((int(query.height*0.8), int(query.width)), Image.ANTIALIAS)
    im1.thumbnail((500,500), Image.ANTIALIAS)
    im2.thumbnail((500,500), Image.ANTIALIAS)
    dst = Image.new('RGB', color="white", size=(im1.width, offset+im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, offset+im1.height))
    return dst
# OUTFIT CONCAT
def get_concat_h(im1, im2):

    urllib.request.urlretrieve(
    im1,
    "im1.jpg")
  
    im1 = Image.open("im1.jpg")

    if isinstance(im1,str):im1 = Image.open(im1)
    if isinstance(im2,str):im2 = Image.open(im2)
    dst = Image.new('RGB',color="white", size=(im1.width + im2.width, im2.height))
    #offset = int((im2.height)*0.2) # start from 20% from top...
    im1.thumbnail((650,650), Image.ANTIALIAS)
    offset=200
    #im1 = im1.resize((int(im2.height*0.8), int(im2.height*0.8)),Image.ANTIALIAS)
    dst = Image.new('RGB',color="white", size=(im1.width + im2.width, im2.height))
    dst.paste(im1, (0, offset))
    dst.paste(im2, (im1.width, 0))
    return dst

#prefix_path_to_imgs = './bluebungalow_2k_single_image'
prefix_path_to_imgs = 'https://static.okkular.io/style-gen/bluebungalow_2k_single_image'
# FIRST OUTFIT
item1 = st.session_state['outfit']['query'] # convert to local format
item1 = item1.split('/')[-1]

item2 = st.session_state['outfit']['item1']['suggestion'][0]
item2 = item2.split('/')[-1]

item3= st.session_state['outfit']['item2']['suggestion'][0]
item3 = item3.split('/')[-1]

v= get_concat_v(f'{prefix_path_to_imgs}/{item2}',f'{prefix_path_to_imgs}/{item3}',f'{prefix_path_to_imgs}/{item1}')
o1 = get_concat_h(f'{prefix_path_to_imgs}/{item1}', v)
col1.write(f'First suggestion, query: {st.session_state["query_category"]}')
col1.image(o1)#, use_column_width=True)

# SECOND OUTFIT
item2 = st.session_state['outfit']['item1']['suggestion'][1]
item2 = item2.split('/')[-1]

item3= st.session_state['outfit']['item2']['suggestion'][1]
item3 = item3.split('/')[-1]

v= get_concat_v(f'{prefix_path_to_imgs}/{item2}',f'{prefix_path_to_imgs}/{item3}', f'{prefix_path_to_imgs}/{item1}')
o2 = get_concat_h(f'{prefix_path_to_imgs}/{item1}', v)
col2.write('Second suggestion')
col2.image(o2)#, use_column_width=True)

# THIRD OUTFIT
col1.markdown("""---""")
item2 = st.session_state['outfit']['item1']['suggestion'][2]
item2 = item2.split('/')[-1]

item3= st.session_state['outfit']['item2']['suggestion'][2]
item3 = item3.split('/')[-1]

v= get_concat_v(f'{prefix_path_to_imgs}/{item2}',f'{prefix_path_to_imgs}/{item3}', f'{prefix_path_to_imgs}/{item1}')
o3 = get_concat_h(f'{prefix_path_to_imgs}/{item1}', v)
col1.write('Third suggestion')
col1.image(o3)#, use_column_width=True)

# FOURTH OUTFIT
col2.markdown("""---""")
item2 = st.session_state['outfit']['item1']['suggestion'][3]
item2 = item2.split('/')[-1]

item3= st.session_state['outfit']['item2']['suggestion'][3]
item3 = item3.split('/')[-1]

v= get_concat_v(f'{prefix_path_to_imgs}/{item2}',f'{prefix_path_to_imgs}/{item3}', f'{prefix_path_to_imgs}/{item1}')
o3 = get_concat_h(f'{prefix_path_to_imgs}/{item1}', v)
col2.write('Fourth suggestion')
col2.image(o3)#, use_column_width=True)



# CSS FOR BORDERS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

local_css("styles.css")
