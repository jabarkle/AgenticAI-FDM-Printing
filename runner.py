from chain import *
import json
from utils import *
import requests
import os
import json
import time
import cv2
from loguru import logger
from langchain_core.callbacks import FileCallbackHandler, StdOutCallbackHandler
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from image_inference import *
import glob2

from PIL import Image, ImageDraw, ImageFont
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

url = "172.26.197.215:81"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

os.environ["QT_QPA_PLATFORM"] = "offscreen"

def get_toolhead_state(url,debug=False):
    """
    Fetches the state of the toolhead from a Moonraker 3D printer.

    Args:
    url (str): The base URL to the Moonraker API.
    """
    full_url = f"http://{url}/printer/objects/query?toolhead"
    try:
        # Send a request to the Moonraker API to get the state of the toolhead
        response = requests.get(full_url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the JSON response and extract toolhead status
        data = response.json()
        toolhead_state = data.get('result', {}).get('status', {}).get('toolhead', {})

        # Print the state of the toolhead
        if debug:
            print("Toolhead State:")
            for key, value in toolhead_state.items():
                print(f"{key}: {value}")

        return True, toolhead_state

    except requests.RequestException as e:
        print(f"Failed to retrieve toolhead state: {e}")

        return False, e
    
def crop_combine(top_loc, front_loc, save_loc):
    img = Image.open(top_loc)
    img.save(save_loc)
    return



def resume_print(ip_address):
    """
    Pause the current print job on a 3D printer managed by MainsailOS.

    Args:
    ip_address (str): The IP address of the MainsailOS server.
    """
    url = f"http://{ip_address}/printer/print/resume"
    headers = {'Content-Type': 'application/json'}
    data = {}  # Depending on your setup, this might need to be 'M25' or another specific command

    # try:
    response = requests.post(url, json=data, headers=headers)



def get_printer_state(url):
    """
    Fetches the state of the printer from a Moonraker 3D printer.
    
    Args:
    url (str): The base URL to the Moonraker API.
    """
    url = f"http://{url}/printer/objects/query?print_stats"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data

def get_timelapse_image(frame_no, path):
    """
    Fetches a timelapse image from a Moonraker 3D printer.
    
    Args:
    frame_no (int): The frame number of the timelapse image to fetch.
    """
    url = f"http://172.26.197.215:81/server/files/timelapse_frames/frame{frame_no}.jpg"
    response = requests.get(url)
    response.raise_for_status()
    with open(path, 'wb') as f:
        f.write(response.content)


def get_image(url, path, mode="front"):
    """gets the image of the printer encodes it"""
    print(f"Getting {mode} image")
    url = f"http://{url}/webcam/?action=snapshot"
    response = requests.get(url)
    if response.status_code == 200:
        image_path = path
        with open(image_path, "wb") as f:
            f.write(response.content)
        # print("Image saved as printer_image.jpg")
    else:
        print("Failed to retrieve image, status code:", response.status_code)



def check_previous_images(path):
    img_list = glob2.glob(path+"/top_**.jpg")

    num=[0]
    for i in img_list:
        num.append(int(i.split("_")[-1].split(".")[0]))

    return max(num)


def runner(printer_url="172.26.197.215:81",
           image_save_path="./results/videotest/layer_images",
           save_path="./results/videotest",
           image_resize_factor=2,
           openai_api_key=OPENAI_API_KEY,
           ):

    flag= True

    # prompts = json.load(open("./prompts/system_prompt.json", "r"))
    image_system_prompt=load_text_file("./prompts/image_system_prompts.txt")
    image_user_prompt=load_text_file("./prompts/image_user_prompt.txt")
    

    while flag:

        max_tries = 5
        print(f"\033[91mChecking printer state\033[0m")

        current_state = get_printer_state(url)
        # print(current_state)
            
        printer_status = current_state["result"]["status"]["print_stats"]["state"]
        current_layer = current_state["result"]["status"]["print_stats"]["info"]["current_layer"]

        print(f"\033[91mPrinter Status: {printer_status}\033[0m")
        print(f"\033[91mCurrent Layer: {current_layer}\033[0m")

        if printer_status == "printing":
            time.sleep(20)

        if printer_status == "complete":
            print("\033[91mPrinting Complete. Exiting.\033[0m")
            flag=False

        
        if printer_status == "paused":
            print("\033[91mPrinter is paused. Take snapshot.\033[0m")
            time.sleep(10)

            # nn = check_previous_images(image_save_path)
            nn = current_layer
            # nn=nn+1
            pre=2
            if nn>=3:
                pre=nn-1

            # print(nn)
        
            # image_top = get_timelapse_image(str("%06d" % nn), image_save_path+f"/top_layer_{nn}.jpg")
            image_front = get_image(printer_url, image_save_path+ f"/front_layer_{nn}.jpg")
            image_top = get_image(printer_url, image_save_path+ f"/top_layer_{nn}.jpg", mode="top")

            print("Image saved.")
            time.sleep(5)

            # croped = crop_combine(top_loc=image_save_path+ f"/top_layer_{nn}.jpg", front_loc=image_save_path+ f"/front_layer_{nn}.jpg")
        
            # prompts = json.load(open("./prompts/system_prompt.json", "r"))
            image_system_prompt=load_text_file("./prompts/image_system_prompts.txt")
            image_user_prompt=load_text_file("./prompts/image_user_prompt.txt")

            # failures,r = send_image(image_save_path+f"/top_layer_{nn}.jpg",system_prompt=prompts["system_prompt_eyes"],user_prompt=f"This the current image at layer {current_layer}. Identify the most visually prominent defects in the current layer.", resize_factor=image_resize_factor, api_key=openai_api_key, image_path_2=image_save_path+f"/front_layer_{pre}.jpg")

            croped = crop_combine(top_loc=image_save_path+ f"/top_layer_{nn}.jpg", front_loc=image_save_path+ f"/front_layer_{nn}.jpg", save_loc=image_save_path+ f"/combined_{nn}.jpg")

            previous_combined = image_save_path + f"/combined_{pre}.jpg"
            if os.path.exists(previous_combined) and pre != nn:
                print("Running image inference with previous layer reference")
                failures = send_image(image_save_path+f"/combined_{nn}.jpg", system_prompt=image_system_prompt, user_prompt = image_user_prompt, resize_factor=2, previous_image_path=previous_combined)
            else:
                print("Running image inference (first layer, no previous reference)")
                failures = send_image(image_save_path+f"/combined_{nn}.jpg", system_prompt=image_system_prompt, user_prompt = image_user_prompt, resize_factor=2)

            #save failures to a file
            with open(save_path+f"/failures_{nn}.txt", "w") as f:
                f.write(failures.content)

            print(f"\033[92m Detected Failures:\n {failures}\033[0m")

            #check if a file exists
            if os.path.exists(save_path+f"/previous_solution_{pre}.txt"):
                previous_solution = load_text_file(save_path+f"/previous_solution_{pre}.txt")
            else:
                previous_solution = []

            graph = get_graph()

            print("\033[94mRunning LLM AGENT")

            logfile = save_path+f"/log{nn}.log"
            
            logger.add(logfile, colorize=True, enqueue=True)
            handler_1 = FileCallbackHandler(logfile)
            handler_2 = StdOutCallbackHandler()
            
            reasoning_planner=load_text_file("./prompts/info_reasoning.txt")
            # observation=load_text_file("failure.txt")
            observation = failures.content  
            printer_objects = load_text_file("./prompts/printer_objects.txt")
            solution_reasoning= load_text_file("./prompts/solution_reasoning.txt")
            gcode_cmd = load_text_file("./prompts/gcode_commands.txt")

            out = graph.invoke(
                {
                    "internal_messages": ["Given the failure (if any) plan for what information is required to identify the issue, query printer for the required information, plan the solution steps to solve the problem, execute the solution plan, resume print and finish.\n If no problem is detected, resume print."],

                    "printer_url": printer_url,

                    "information_known":[f"Printer url is http://{printer_url}","Filament type is PLA","printer model is Creality Ender V3 SE", f"Printer status is paused", f"Current layer is {current_layer}", "Tool position is at the home position", "BED is perfectly Calibrated", "Nozzle diameter is 0.4mm", "There are no issues with the nozzle and printer hardware.", "Layer height is 0.2", "Infill pattern is aligned rectilinear"],
                    "observations": observation,
                    "reasoning": reasoning_planner,
                    "solution_reasoning": solution_reasoning, 
                    "printer_obj": printer_objects,
                    "adapted_recon_reasoning" : [],
                    "adapter_solution_reasoning" : [],
                    "gcode_commands":gcode_cmd,
                    "previous_solution" : previous_solution,
                    "solution_steps":[],


                },
                {"callbacks":[handler_1, handler_2]},
                debug=True
                
            )

            previous_solution = out["solution_steps"]
            with open(save_path+f"/previous_solution_{nn}.txt", "w") as f:
                f.write(str(previous_solution))

            with open(save_path+f"/LLM_out_{nn}.txt", "w") as f:
                f.write(str(out))

                

                
runner()
