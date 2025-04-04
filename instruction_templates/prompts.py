ID_format = "<OBJ{:02}>"

scanrefer_prompt = [
    "Please share the ID of the object that closely matches the description <description>."
    "Provide the ID of the object that best fits the description <description>."
    "What is the ID of the object that aligns with the description <description>?"
    "Identify the ID of the object that closely resembles the description <description>."
    "What's the ID of the object that corresponds to the description <description>?"
    "Give the ID of the object that most accurately matches the description <description>."
    "Share the ID of the object that best corresponds to the description <description>."
    "Identify the ID of the object that closely aligns with the description <description>."
]

multi3dref_prompt = [
    "Are there any objects fitting the description <description>? If so, please provide their IDs."
    "Do any objects match the description <description>? If they do, share their IDs."
    "Is there anything that matches the description <description>? If yes, provide their IDs."
    "Are there objects that correspond to the description <description>? If so, kindly list their IDs."
    "Does anything fit the description <description>? If it does, list the IDs of those objects."
    "Have any objects been described as <description>? If so, share their IDs."
    "Do any objects meet the criteria of <description>? If they do, kindly provide their IDs."
    "Are there any objects that correspond to the description <description>? If yes, share their IDs."
]

scan2cap_prompt = [
    "Start by detailing the visual aspects of the <OBJ_ID>, then delve into its spatial context within the scene."
    "Outline the appearance of the <OBJ_ID>, then elaborate on its positioning relative to other objects in the scene."
    "Illustrate the visual attributes of the <OBJ_ID>, then explore its spatial relationships with other elements in the scene."
    "Begin by articulating the outward features of the <OBJ_ID>, then discuss its spatial alignment within the broader scene."
    "Provide a detailed description of the <OBJ_ID>'s appearance before analyzing its spatial connections with other elements in the scene."
    "Capture the essence of the <OBJ_ID>'s appearance, then analyze its spatial relationships within the scene's context."
    "Detail the physical characteristics of the <OBJ_ID>, then examine its spatial dynamics among other objects in the scene."
]

scanqa_prompt = [
    "<Raw Question> Please answer the question using a single word or phrase."
]

scene_descriptions_prompt = [
    "Provide a valid description of the entire scene."
]

obj_caption_wid_prompt = [
    "Portray the visual characteristics of the <id>.",
    "Detail the outward presentation of the <id>.",
    "Provide a depiction of the <id>'s appearance.",
    "Illustrate how the <id> looks.",
    "Describe the visual aspects of the <id>.",
    "Convey the physical attributes of the <id>.",
    "Outline the external features of the <id>.",
    "Render the appearance of the <id> in words.",
    "Depict the outward form of the <id>.",
    "Elaborate on the visual representation of the <id>."
]