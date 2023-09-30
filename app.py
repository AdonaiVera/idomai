"""
Main module to execute models in local cameras.
"""
import streamlit as st
import logging
import logging.handlers
import queue
import urllib.request
import av
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
v_tf = tf.__version__.split('.')[0]
if v_tf == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
from skimage import measure
from random import randint

from pathlib import Path
from typing import List, NamedTuple
from aiortc.contrib.media import MediaPlayer
from PIL import Image
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  

from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

## Changes Names and Icon
icon = Image.open('img/icon.png')
st.set_page_config(page_title = "Idom+ AI", page_icon = icon)


## Inicializate share variables.
HERE = Path(__file__).parent
logger = logging.getLogger(__name__)
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": True},
)

## Main menu.
def main():
    st.header("Academia virtual con Inteligencia Artificial")

    videoFiltersPage = "Aplica filtros con IA"
    smartBoard = "Tablero Inteligente"
    background= "Cambiar fondo"
    loopbackPage = "Video sin filtros"
    app_mode = st.sidebar.selectbox(
        "Selecciona la herramienta que deseas utilizar",
        [
            loopbackPage,
            videoFiltersPage,
            smartBoard,
            background
        ],
    )
    st.subheader(app_mode)

    if app_mode == videoFiltersPage:
        appVideoFilters()
    elif app_mode == loopbackPage:
        appLoopback()
    elif app_mode == smartBoard:
        appSmartBoard()
    elif app_mode == background:
        appBackground()


## Simple video loopback
def appLoopback():
    webrtc_streamer(
        key="loopback",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=None,  
    )

## Comming soon
def appSmartBoard():
    """ Video transforms with OpenCV """

    class MediaPipeTransformer(VideoTransformerBase):
        type: Literal["Manos", "Tablero"]


        ## Inicializate share variables.
        def __init__(self):
            # General configuration
            self.mp_drawing = mp.solutions.drawing_utils
            self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

            # Configuration Hands
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
        
            self.drawPoints = []



        def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = self.hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                (imgH, imgW) = image.shape[:2]
                if self.type == "Manos":
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                elif self.type == "Tablero":               
                    for handLandmarks in results.multi_hand_landmarks:
                        point  = 4
                        normalizedLandmark = handLandmarks.landmark[point]
                        pixelCoordinatesLandmarkTip = self.mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imgW, imgH)
                        point  = 8
                        normalizedLandmark = handLandmarks.landmark[point]
                        pixelCoordinatesLandmarkFingerTip = self.mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imgW, imgH)
                        
                        if type(pixelCoordinatesLandmarkTip) == tuple and type(pixelCoordinatesLandmarkFingerTip) == tuple: 
                            dist = np.linalg.norm(np.asarray(pixelCoordinatesLandmarkTip)-np.asarray(pixelCoordinatesLandmarkFingerTip))
                            if dist < 20:
                                self.drawPoints.append(pixelCoordinatesLandmarkTip)
                                cv2.circle(image, pixelCoordinatesLandmarkTip, 4, (166, 0, 163), cv2.FILLED)
                        
                        cv2.circle(image, pixelCoordinatesLandmarkTip, 10, (0, 255, 0), -1)
                        cv2.circle(image, pixelCoordinatesLandmarkFingerTip, 10, (0, 255, 0), -1)

                    for x in range(0,len(self.drawPoints)):
                        cv2.circle(image, self.drawPoints[x], 4, (166, 0, 163), cv2.FILLED)

            return image
    
               
    webrtc_ctx = webrtc_streamer(
        key="mediaPipe-filter",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_factory=MediaPipeTransformer,
        async_processing=False,
    )


    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.type = st.radio(
            "Selecciona la función que deseas usar", ("Manos", "Tablero")
        )




## Run Model to apply filters to the local images.
def appVideoFilters():
    """ Video transforms with OpenCV """

    class MediaPipeTransformer(VideoTransformerBase):
        type: Literal["mapa puntos", "bigote", "gafas", "sombrero", "anonymous", "monstruo", "oxigeno1", "Mascara Covid 1", "Mascara Covid 2", "Mascara Covid 3", "Mascara Covid 4", "Mascara Covid 5", "Mascara Covid 6", "Mascara Covid 7"]

        ## Inicializate share variables.
        def __init__(self):
            # General configuration
            self.mp_drawing = mp.solutions.drawing_utils
            self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

            # Configuration Face Mesh
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
            

            # Configuration Hands
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)

            # Configuration Pose
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)


            # Import allfilters
            self.filter1 = cv2.imread('filters/filter1.png', cv2.IMREAD_UNCHANGED)
            self.filter2 = cv2.imread('filters/filter2.png', cv2.IMREAD_UNCHANGED)
            self.filter3 = cv2.imread('filters/filter3.png', cv2.IMREAD_UNCHANGED)
            self.filter4 = cv2.imread('filters/filter4.png', cv2.IMREAD_UNCHANGED)
            self.filter5 = cv2.imread('filters/filter5.png', cv2.IMREAD_UNCHANGED)
            self.filter6 = cv2.imread('filters/filter6.png', cv2.IMREAD_UNCHANGED)
            self.filter8 = cv2.imread('filters/filter8.png', cv2.IMREAD_UNCHANGED)
            self.filter9 = cv2.imread('filters/filter9.png', cv2.IMREAD_UNCHANGED)
            self.filter10 = cv2.imread('filters/filter10.png', cv2.IMREAD_UNCHANGED)
            self.filter11 = cv2.imread('filters/filter11.png', cv2.IMREAD_UNCHANGED)
            self.filter12 = cv2.imread('filters/filter12.png', cv2.IMREAD_UNCHANGED)
            self.filter13 = cv2.imread('filters/filter13.png', cv2.IMREAD_UNCHANGED)
            self.filter14 = cv2.imread('filters/filter14.png', cv2.IMREAD_UNCHANGED)


        '''
        Class to apply Mash filter to the image
        @param image with out filter
        @return image with filter
        '''
        def classMesh(self, image):
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = self.face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
     
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:

                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACE_CONNECTIONS,
                        landmark_drawing_spec=self.drawing_spec,
                        connection_drawing_spec=self.drawing_spec)

            return image

        
        '''
        Class to apply add a full mask filter
        @param original image, filter, x, y, width and face height
        @return image with filter
        '''
        def applyFilter(self, source, imageFace, dstMat):
            (imgH, imgW) = imageFace.shape[:2]

            #filterImg = cv2.resize(filterImg,(face_width,face_height))
            # grab the spatial dimensions of the source image and define the
            # transform matrix for the *source* image in top-left, top-right,
            # bottom-right, and bottom-left order
            (srcH, srcW) = source.shape[:2]          
            srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])

            # compute the homography matrix and then warp the source image to the
            # destination based on the homography
            (H, _) = cv2.findHomography(srcMat, dstMat)
            warped = cv2.warpPerspective(source, H, (imgW, imgH))


            # Split out the transparency mask from the colour info
            overlay_img = warped[:,:,:3] # Grab the BRG planes
            overlay_mask = warped[:,:,3:]  # And the alpha plane
        
            # Again calculate the inverse mask
            background_mask = 255 - overlay_mask
        
            # Turn the masks into three channel, so we can use them as weights
            overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
            background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
        
            # Create a masked out face image, and masked out overlay
            # We convert the images to floating point in range 0.0 - 1.0
            face_part = (imageFace * (1 / 255.0)) * (background_mask * (1 / 255.0))
            overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))
            output = np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))
            return output
    

    

        '''
        Class to get points
        @param results points of IA, real image, and type of filter
        @return matriz de destino del filtro.
        '''
        def classPoints(self, results, image, typeFilter):
            if typeFilter == "completeFace":
                # Transform matrix, making sure the points are specified in 
                # top-left, top-right, bottom-right, and bottom-left order.
                ## Multiplying a scalar.
                topLeft = [float(results[0].landmark[54].x * image.shape[1]) - (float(results[0].landmark[54].x * image.shape[1])*0.10), float(results[0].landmark[54].y * image.shape[0]) - (float(results[0].landmark[54].y * image.shape[0]))*0.10]
                topRight = [float(results[0].landmark[284].x * image.shape[1]) + (float(results[0].landmark[284].x * image.shape[1])*0.10) , float(results[0].landmark[284].y * image.shape[0]) - (float(results[0].landmark[284].y * image.shape[0])*0.10)]
                bottomRight = [float(results[0].landmark[365].x * image.shape[1])+ (float(results[0].landmark[365].x * image.shape[1])*0.10) , float(results[0].landmark[365].y * image.shape[0]) + (float(results[0].landmark[365].y * image.shape[0])*0.10)]
                bottomLeft = [float(results[0].landmark[136].x * image.shape[1]) - (float(results[0].landmark[136].x * image.shape[1])*0.10) , float(results[0].landmark[136].y * image.shape[0]) + (float(results[0].landmark[136].y * image.shape[0])*0.10)]

            elif typeFilter == "mask":
                ## Taking Real points
                topLeft = [float(results[0].landmark[127].x * image.shape[1]), float(results[0].landmark[127].y * image.shape[0])]
                topRight = [float(results[0].landmark[356].x * image.shape[1]) , float(results[0].landmark[356].y * image.shape[0])]
                bottomRight = [float(results[0].landmark[365].x * image.shape[1]) , float(results[0].landmark[152].y * image.shape[0])]
                bottomLeft = [float(results[0].landmark[136].x * image.shape[1]) , float(results[0].landmark[152].y * image.shape[0])]

            elif typeFilter == "bigote":
                ## Taking Real points
                topLeft = [float(results[0].landmark[205].x * image.shape[1]), float(results[0].landmark[205].y * image.shape[0])]
                topRight = [float(results[0].landmark[425].x * image.shape[1]) , float(results[0].landmark[425].y * image.shape[0])]
                bottomRight = [float(results[0].landmark[436].x * image.shape[1]) , float(results[0].landmark[436].y * image.shape[0])]
                bottomLeft = [float(results[0].landmark[216].x * image.shape[1]) , float(results[0].landmark[216].y * image.shape[0])]


            elif typeFilter == "eyes":
                ## Taking Real points
                topLeft = [float(results[0].landmark[21].x * image.shape[1]), float(results[0].landmark[21].y * image.shape[0])]
                topRight = [float(results[0].landmark[251].x * image.shape[1]) , float(results[0].landmark[251].y * image.shape[0])]
                bottomRight = [float(results[0].landmark[323].x * image.shape[1]) , float(results[0].landmark[323].y * image.shape[0])]
                bottomLeft = [float(results[0].landmark[93].x * image.shape[1]) , float(results[0].landmark[93].y * image.shape[0])]

            elif typeFilter == "head":
                ## Taking Real points
                topLeft = [float(results[0].landmark[54].x * image.shape[1]) , float(results[0].landmark[54].y * image.shape[0]) - (float(results[0].landmark[54].y * image.shape[0]))*0.50]
                topRight = [float(results[0].landmark[251].x * image.shape[1]) , float(results[0].landmark[54].y * image.shape[0]) - (float(results[0].landmark[54].y * image.shape[0])*0.50)]
                bottomRight = [float(results[0].landmark[251].x * image.shape[1]) , float(results[0].landmark[251].y * image.shape[0])]
                bottomLeft = [float(results[0].landmark[54].x * image.shape[1]) , float(results[0].landmark[54].y * image.shape[0])]


            dstMat = [ topLeft, topRight, bottomRight, bottomLeft ]
            dstMat = np.array(dstMat)
            return dstMat


        '''
        Class to apply Pose filter to the image
        @param image with out filter
        @return image with filter
        '''
        def classPose(self, image):
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = self.pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            return image

        def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
            try:
                img = frame.to_ndarray(format="bgr24")

                image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = self.face_mesh.process(image)

                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                resultImg = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                
                ## Sorry for so many conditionals, I must optimize this part
                if results.multi_face_landmarks:
                    if self.type != "mapa puntos":
                        if self.type == "anonymous":
                            typeFilter = "completeFace"
                            filterImg = self.filter1
                        
                        elif self.type == "bigote":
                            typeFilter = "bigote"
                            filterImg = self.filter2

                        elif self.type == "gafas":
                            typeFilter = "eyes"
                            filterImg = self.filter3

                        elif self.type == "sombrero":
                            typeFilter = "head"
                            filterImg = self.filter4

                        elif self.type == "monstruo":
                            typeFilter = "completeFace"
                            filterImg = self.filter5

                        elif self.type == "oxigeno1":
                            typeFilter = "completeFace"
                            filterImg = self.filter6

                        elif self.type == "Mascara Covid 1":
                            typeFilter = "mask"
                            filterImg = self.filter8

                        elif self.type == "Mascara Covid 2":
                            typeFilter = "mask"
                            filterImg = self.filter9

                        elif self.type == "Mascara Covid 3":
                            typeFilter = "mask"
                            filterImg = self.filter10

                        elif self.type == "Mascara Covid 4":
                            typeFilter = "mask"
                            filterImg = self.filter11

                        elif self.type == "Mascara Covid 5":
                            typeFilter = "mask"
                            filterImg = self.filter12

                        elif self.type == "Mascara Covid 6":
                            typeFilter = "mask"
                            filterImg = self.filter13

                        elif self.type == "Mascara Covid 7":
                            typeFilter = "mask"
                            filterImg = self.filter14

                        dstMat = self.classPoints(results.multi_face_landmarks, image, typeFilter) 
                        resultImg = self.applyFilter(filterImg,resultImg,dstMat)  

                        
                    else:
                        for face_landmarks in results.multi_face_landmarks:
                            self.mp_drawing.draw_landmarks(
                                image=resultImg,
                                landmark_list=face_landmarks,
                                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=self.drawing_spec,
                                connection_drawing_spec=self.drawing_spec)
                            
            except Exception as e:
                print(e)
                pass
            
            return resultImg


    webrtc_ctx = webrtc_streamer(
        key="mediaPipe-filter",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_factory=MediaPipeTransformer,
        async_transform=False,
    )

    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.type = st.radio(
            "Selecciona el tipo de filtro que deseas aplicar", ("mapa puntos", "bigote", "gafas", "sombrero", "anonymous", "monstruo", "oxigeno1", "Mascara Covid 1", "Mascara Covid 2", "Mascara Covid 3", "Mascara Covid 4", "Mascara Covid 5", "Mascara Covid 6", "Mascara Covid 7")
        )


## Run Model to apply filters to the local images.
def appBackground():
    """ Video transforms with OpenCV """

    class MediaPipeTransformer(VideoTransformerBase):
        type: Literal["Palacio de la Inquisición", "Torre del Reloj", "Santuario de San Pedro Claver", "Cuartel de las Bóvedas", "Convento de Santa Cruz de la Popa"]

        ## Inicializate share variables.
        def __init__(self):
            self.graph = self.load_model()
            self.config = tf.ConfigProto(allow_soft_placement=True)
            self.config.gpu_options.allow_growth = True
            
            self.graph.as_default()
            self.sess = tf.Session(graph=self.graph)
            self.target_size = (513, 384)
            self.background1 = cv2.imread('background/background1.jpg')
            self.background2 = cv2.imread('background/background2.jpg')
            self.background3 = cv2.imread('background/background3.jpg')
            self.background4 = cv2.imread('background/background4.jpg')
            self.background5 = cv2.imread('background/background5.jpg')
            self.background1 = cv2.resize( self.background1, self.target_size)
            self.background2 = cv2.resize( self.background2, self.target_size)
            self.background3 = cv2.resize( self.background3, self.target_size)
            self.background4 = cv2.resize( self.background4, self.target_size)
            self.background5 = cv2.resize( self.background5, self.target_size)

    

        '''
        Class to apply Pose filter to the image
        @param image with out filter
        @return image with filter
        '''
        def classPose(self, image):
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = self.pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            return image

        '''
        Class to load the graph in TensorFlow
        @param 
        @return graph
        '''
        def load_model(self):
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                seg_graph_def = tf.GraphDef()
                with tf.gfile.GFile('models/frozen_inference_graph_small.pb', 'rb') as fid:
                    serialized_graph = fid.read()
                    seg_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(seg_graph_def, name='')
            return detection_graph

        '''
        Read the image and apply filteres
        '''
        def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            image = cv2.resize(img, self.target_size)
            batch_seg_map = self.sess.run('SemanticPredictions:0',
                                        feed_dict={'ImageTensor:0': [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]})
            # visualization
            seg_map = batch_seg_map[0]
            seg_map[seg_map != 15] = 0
           
            ## Sorry for so many conditionals, I must optimize this part
            if self.type == "Palacio de la Inquisición":
                bg_copy=self.background1.copy()
    
            elif self.type == "Torre del Reloj":
                bg_copy=self.background2.copy()

            elif self.type == "Santuario de San Pedro Claver":
                bg_copy=self.background3.copy()

            elif self.type == "Cuartel de las Bóvedas":
                bg_copy=self.background4.copy()

            elif self.type == "Convento de Santa Cruz de la Popa":
                bg_copy=self.background5.copy()

            mask = (seg_map == 15)
            bg_copy[mask] = image[mask]

            # create_colormap(seg_map).astype(np.uint8)
            seg_image = np.stack(
                (seg_map, seg_map, seg_map), axis=-1).astype(np.uint8)
            gray = cv2.cvtColor(seg_image, cv2.COLOR_BGR2GRAY)

            thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]

            cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)

            try:
                cv2.drawContours(
                    bg_copy, cnts, -1, (randint(0, 255), randint(0, 255), randint(0, 255)), 2)
            except:
                pass             
                
            return bg_copy

    webrtc_ctx = webrtc_streamer(
        key="mediaPipe-filter",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_factory=MediaPipeTransformer,
        async_transform=False,
    )


    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.type = st.radio(
            "Selecciona el tipo de filtro que deseas aplicar", ("Palacio de la Inquisición", "Torre del Reloj", "Santuario de San Pedro Claver", "Cuartel de las Bóvedas", "Convento de Santa Cruz de la Popa")
        )


## Main
if __name__ == "__main__":
    # Basic configuration of the app
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG)

    ## Saving events in case of bugs 
    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()