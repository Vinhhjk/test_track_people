CAMERA_URL = 0
CACHE_UPDATE_INTERVAL = 30  # seconds
ABSENCE_THRESHOLD = 10 # seconds

MODEL_PATHS = {
    "face_detector": ("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"),
    "age_net": ("models/age_deploy.prototxt", "models/age_net.caffemodel"),
    "gender_net": ("models/gender_deploy.prototxt", "models/gender_net.caffemodel"),
}

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-22)','(23-24)','(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

