import os

import bpy

# DATA:
scale_factor = 0.07
output_path = "resized_facescape"
DATA_DIR = "./facescape_trainset_001_100"
expressions_name = ["1_neutral", "2_smile", "3_mouth_stretch", '4_anger', "5_jaw_left", "6_jaw_right",
                    "7_jaw_forward", "8_mouth_left", "9_mouth_right", "10_dimpler", "11_chin_raiser",
                    "12_lip_puckerer", "13_lip_funneler", "14_sadness", "15_lip_roll", "16_grin",
                    "17_cheek_blowing", "18_eye_closed", "19_brow_raiser", "20_brow_lower"]

for idx in range(100):

    obj_folder = os.path.join(DATA_DIR, str(idx + 1) + "/models_reg")
    output_folder = os.path.join(output_path, str(idx + 1) + "/models_reg")
    for name in expressions_name:
        obj_name = os.path.join(obj_folder, name + ".obj")
        output_name = os.path.join(output_folder, name + ".obj")
        # Import the .obj file
        try:
            bpy.ops.import_scene.obj(filepath=obj_name)
        except:
            continue
        # Get the newly imported object
        imported_object = bpy.context.selected_objects[0]
        # Scale the imported object
        imported_object.scale *= scale_factor
        # Export the modified object as .obj file
        bpy.ops.export_scene.obj(filepath=output_name, use_selection=True)
        # Deselect all objects to prepare for the next import
        bpy.ops.object.select_all(action='DESELECT')


print("Object resized and saved successfully!")
