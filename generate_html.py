def get_picture_html(path, tag):
    image_html = """<p> {tag_name} </p> <picture> <img src= "../{path_name}"  height="300" width="400"> </picture>"""
    return image_html.format(tag_name=tag, path_name=path)


# define function to add the list element in the html file
def get_count_html(category, count):
    count_html = """<li> {category_name} : {count_} </li>"""
    return count_html.format(category_name=category, count_=count)


# function to calculate the value count
def get_value_count(image_class_dict):
    count_dic = {}
    for category in image_class_dict.values():
        if category in count_dic.keys():
            count_dic[category] = count_dic[category] + 1
        else:
            count_dic[category] = 1
    return count_dic


# function to generate the html file from image_class dictionary
# keys will be the path of the images and values will be the class associated to it.
def generate_html(image_class_dict):
    picture_html = ""
    count_html = ""

    # loop through the keys and add image to the html file
    for image in image_class_dict.keys():
        picture_html += get_picture_html(path=image, tag=image_class_dict[image])

    value_counts = get_value_count(image_class_dict)

    # loop through the value_counts and add a count of class to the html file
    for value in value_counts.keys():
        count_html += get_count_html(value, value_counts[value])