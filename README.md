# IDOM+ AI  🤓
Teaching Academy with AI

Sample page hosted on Heroku: tryit ... https://idomai.herokuapp.com/ 📢


# Getting Started 🎁
1. Clone repository: For security use SSH keys.
   ```shell
   git clone https://github.com/AdonaiVera/idomai.git
   ```

2. Install dependencies.
   ```shell
   pip3 install -r requirements.txt
   ```

3. Run commands
   ```shell
   streamlit run app.py
   ```

3. Enjoy

# Results 🛠️

### MediaPipe
<div align="center">
       <img src="https://github.com/AdonaiVera/idomai/blob/master/img/fondo.png" width="400px"</img> 
</div>


### Media Pipe with filters
![https://github.com/AdonaiVera/idomai/blob/master/img/filter1.png](https://github.com/AdonaiVera/idomai/blob/master/img/filter1.PNG)

### Media Pipe hands | Smart Board
![https://github.com/AdonaiVera/idomai/blob/master/img/filter2.png](https://github.com/AdonaiVera/idomai/blob/master/img/filter2.PNG)


# Manual deployment 📦
## Manual deployment to heroku 
### Prerequisites
1. Set up `heroku` command.

2. Add [`heroku-buildpack-apt`](https://github.com/heroku/heroku-buildpack-apt) to buildpacks.
   ```shell
   $ heroku buildpacks:add --index 1 heroku-community/apt
   ```
### Deploy
#### If dependencies have changed, update `requirements.txt`
1. Update `requirements.txt`.
   ```shell
   $ make deps/update
   ```

2. Commit it.
   ```shell
   $ git add requirements.txt
   $ git commit -m "requirements.txt"
   ```
#### Deploy the current branch to Heroku
   ```shell
   git push heroku <current-branch>:main
   ```

## Manual deployment to app services Azure 📦
### Prerequisites

1. Install docker
2. Instal Azure CLI locally

### Deploy
### If codes have changed, you should rebuild the docker images
#### Build
   ```shell
   docker build --tag idomai-app .
   ```

#### Try it locally
   ```shell
   docker run -p 8501:8501 idomai-app
   ```

#### Access locally
   ```shell
   http://localhost:8501/
   ```

### Deploy the docker image and save to Azure container Registry

   ```shell
   az acr build --registry idomAIRegistry --resource-group idomAI --image idomai-app .

   ```

# Architecture 📌
Architecture is divided intro three main parts:
1. WEB APP develop in streamlit.
2. Computer vision class develop in Python with mediaPipe and OpenCV.
3. Azure media services to stream videos in real time

<div align="center">
       <img src="https://github.com/AdonaiVera/idomai/blob/master/img/architecture.png" width="800px"</img> 
</div>


# Build with 🛠️
_Mention the tools you used to create your project_

* [MediaPipe](https://mediapipe.dev/) - ML solutions for live and streaming media.
* [Streamlit](https://www.streamlit.io/) - Web apps framework.
* [AzureMediaSerives](https://azure.microsoft.com/es-es/services/media-services/) - Streaming video platform.
* [OpenCV](https://opencv.org/) - Image Processing library


# Contribute ✒️
* **Adonai Vera** - *AI developer Geta Club* - [AdonaiVera](https://github.com/AdonaiVera)


