import re
import numpy as np
import cv2 as cv
from base64 import b64decode, b64encode
import PIL
import io

JS_CODE = '''
    var video;
    var div = null;
    var stream;
    var captureCanvas;
    var imgElement;
    var labelElement;
    
    var pendingResolve = null;
    var shutdown = false;
    
    function removeDom() {
       stream.getVideoTracks()[0].stop();
       video.remove();
       div.remove();
       video = null;
       div = null;
       stream = null;
       imgElement = null;
       captureCanvas = null;
       labelElement = null;
    }
    
    function onAnimationFrame() {
      if (!shutdown) {
        window.requestAnimationFrame(onAnimationFrame);
      }
      if (pendingResolve) {
        var result = "";
        if (!shutdown) {
          captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
          result = captureCanvas.toDataURL('image/jpeg', 0.8)
        }
        var lp = pendingResolve;
        pendingResolve = null;
        lp(result);
      }
    }
    
    async function createDom() {
      if (div !== null) {
        return stream;
      }

      div = document.createElement('div');
      div.style.border = '2px solid black';
      div.style.padding = '3px';
      div.style.width = '100%';
      div.style.maxWidth = '600px';
      document.body.appendChild(div);
      
      const modelOut = document.createElement('div');
      modelOut.innerHTML = "<span>Status:</span>";
      labelElement = document.createElement('span');
      labelElement.innerText = 'No data';
      labelElement.style.fontWeight = 'bold';
      modelOut.appendChild(labelElement);
      div.appendChild(modelOut);
           
      video = document.createElement('video');
      video.style.display = 'block';
      video.width = div.clientWidth - 6;
      video.setAttribute('playsinline', '');
      video.onclick = () => { shutdown = true; };
      stream = await navigator.mediaDevices.getUserMedia(
          {video: { facingMode: "environment"}});
      div.appendChild(video);

      imgElement = document.createElement('img');
      imgElement.style.position = 'absolute';
      imgElement.style.zIndex = 1;
      imgElement.onclick = () => { shutdown = true; };
      div.appendChild(imgElement);
      
      const instruction = document.createElement('div');
      instruction.innerHTML = 
          '<span style="color: red; font-weight: bold;">' +
          'When finished, click here or on the video to stop this demo</span>';
      div.appendChild(instruction);
      instruction.onclick = () => { shutdown = true; };
      
      video.srcObject = stream;
      await video.play();

      captureCanvas = document.createElement('canvas');
      captureCanvas.width = 640; //video.videoWidth;
      captureCanvas.height = 480; //video.videoHeight;
      window.requestAnimationFrame(onAnimationFrame);
      
      return stream;
    }
    async function stream_frame(label, imgData) {
      if (shutdown) {
        removeDom();
        shutdown = false;
        return '';
      }

      var preCreate = Date.now();
      stream = await createDom();
      
      var preShow = Date.now();
      if (label != "") {
        labelElement.innerHTML = label;
      }
            
      if (imgData != "") {
        var videoRect = video.getClientRects()[0];
        imgElement.style.top = videoRect.top + "px";
        imgElement.style.left = videoRect.left + "px";
        imgElement.style.width = videoRect.width + "px";
        imgElement.style.height = videoRect.height + "px";
        imgElement.src = imgData;
      }
      
      var preCapture = Date.now();
      var result = await new Promise(function(resolve, reject) {
        pendingResolve = resolve;
      });
      shutdown = false;
      
      return {'create': preShow - preCreate, 
              'show': preCapture - preShow, 
              'capture': Date.now() - preCapture,
              'img': result};
    }
    '''


# function to convert the JavaScript object into an OpenCV image
def js_to_image(js_reply):
  """
  Params:
          js_reply: JavaScript object containing image from webcam
  Returns:
          img: OpenCV BGR image
  """
  # decode base64 image
  image_bytes = b64decode(js_reply.split(',')[1])
  # convert bytes to numpy array
  jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
  # decode numpy array into OpenCV BGR image
  img = cv.imdecode(jpg_as_np, flags=1)

  return img

# function to convert OpenCV Rectangle bounding box image into base64 byte string to be overlayed on video stream
def bbox_to_bytes(bbox_array):
  """
  Params:
          bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream.
  Returns:
        bytes: Base64 image byte string
  """
  # convert array into PIL image
  bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGB')
  iobuf = io.BytesIO()
  # format bbox into png for return
  bbox_PIL.save(iobuf, format='png')
  # format return string
  bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))

  return bbox_bytes

def write_config(base_config_path,labelmap_path,train_record_path,test_record_path,fine_tune_checkpoint,batch_size,num_steps):
    config = None
    with open(base_config_path) as f:
        config = f.read()

    with open('model_config.config', 'w') as f:
    
      # Set labelmap path
      config = re.sub('label_map_path: ".*?"', 
                 'label_map_path: "{}"'.format(labelmap_path), config)
    
      # Set fine_tune_checkpoint path
      config = re.sub('fine_tune_checkpoint: ".*?"',
                      'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), config)
    
      # Set train tf-record file path
      config = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 
                      'input_path: "{}"'.format(train_record_path), config)
    
      # Set test tf-record file path
      config = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 
                      'input_path: "{}"'.format(test_record_path), config)
    
      # Set number of classes.
      config = re.sub('num_classes: [0-9]+',
                      'num_classes: {}'.format(15), config)
    
      # Set batch size
      config = re.sub('batch_size: [0-9]+',
                      'batch_size: {}'.format(batch_size), config)
    
      # Set training steps
      config = re.sub('num_steps: [0-9]+',
                      'num_steps: {}'.format(num_steps), config)
    
      # Set fine-tune checkpoint type to detection
      config = re.sub('fine_tune_checkpoint_type: "classification"', 
                 'fine_tune_checkpoint_type: "{}"'.format('detection'), config)
    
      config = re.sub('learning_rate_base: 8e-2', 
                'learning_rate_base: 2e-2', config)
    
     #Optimizer settings
      config = re.sub('''optimizer {
        momentum_optimizer: {
          learning_rate: {
            cosine_decay_learning_rate {
              learning_rate_base: 2e-2
              total_steps: 300000
              warmup_learning_rate: .001
              warmup_steps: 2500
            }
          }
          momentum_optimizer_value: 0.9
        }
        use_moving_average: false
      }''', 
                '''optimizer {
        adam_optimizer: {
          learning_rate: {
            manual_step_learning_rate {
              initial_learning_rate: 0.02
              schedule {
                step: 2000
                learning_rate: .0002
              }
              schedule {
                step: 3600
                learning_rate: .00008
              }
              schedule {
                step: 4600
                learning_rate: .00004
              }
            }
          }
        }
        use_moving_average: false
      }
    ''', config)
    


    # Augmentation settings
      config = re.sub('''   data_augmentation_options {
        random_scale_crop_and_pad_to_square {
          output_size: 640
          scale_min: 0.1
          scale_max: 2.0
        }
      }''', 
                '''  data_augmentation_options {
        random_scale_crop_and_pad_to_square {
          output_size: 640
          scale_min: 0.1
          scale_max: 2.0
        }

    random_adjust_brightness {
      max_delta: 0.2
    }


    random_adjust_contrast {
      min_delta: 0.7
      max_delta: 1.1
    }


    random_adjust_hue {
      max_delta: 0.01
    }


    random_adjust_saturation {
      min_delta: 0.75
      max_delta: 1.15
    }

    random_black_patches {
      max_black_patches: 5
      probability: 0.8
      size_to_image_ratio: 0.08
    }


      }''', config)
    


      f.write(config)