const express = require('express');
const multer = require('multer');
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');
const canvas = require('canvas');

// 1. Setup TensorFlow (CPU Version) BEFORE face-api
// This prevents it from looking for the missing C++ bindings
const tf = require('@tensorflow/tfjs');
const faceapi = require('@vladmandic/face-api');

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();
const PORT = process.env.PORT || 3000;
const upload = multer({ storage: multer.memoryStorage() });

// --- ABSOLUTE PATH SETUP ---
// This finds the exact folder where this script is running
const ROOT_DIR = __dirname;
const MODEL_DIR = path.join(ROOT_DIR, 'models');

async function loadModels() {
    console.log('-----------------------------------');
    console.log(`🔍 Checking for models in: ${MODEL_DIR}`);
    
    // DEBUG: Print file structure to logs
    try {
        if (!fs.existsSync(MODEL_DIR)) {
            console.error('❌ CRITICAL: The "models" folder does not exist on the server!');
            console.log('📂 Current Folder Contents:', fs.readdirSync(ROOT_DIR));
            throw new Error('Missing models folder');
        }
        const files = fs.readdirSync(MODEL_DIR);
        console.log('✅ Found these model files:', files);
    } catch (error) {
        console.error('❌ File System Error:', error.message);
        process.exit(1); // Stop app if files are missing
    }

    console.log('🏗️  Loading Face API (CPU Mode)...');
    try {
        await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_DIR);
        await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_DIR);
        console.log('✅ Models loaded successfully');
    } catch (error) {
        console.error('❌ Model Loading Failed:', error);
        process.exit(1);
    }
    console.log('-----------------------------------');
}

// --- Cropping Logic ---
function calculatePassportCrop(box, imageWidth, imageHeight) {
    const faceCenterX = box.x + (box.width / 2);
    const faceCenterY = box.y + (box.height / 2);

    // Face height should be ~75% of photo
    const targetHeight = box.height / 0.75; 
    const targetWidth = targetHeight * 0.77; // Ratio 3.5 : 4.5

    let cropX = faceCenterX - (targetWidth / 2);
    let cropY = faceCenterY - (targetHeight / 2);
    
    // Shift up slightly for passport framing
    cropY = cropY - (box.height * 0.1); 

    // Boundary Checks
    if (cropX < 0) cropX = 0;
    if (cropY < 0) cropY = 0;
    if (cropX + targetWidth > imageWidth) cropX = imageWidth - targetWidth;
    if (cropY + targetHeight > imageHeight) cropY = imageHeight - targetHeight;

    return {
        left: Math.round(cropX),
        top: Math.round(cropY),
        width: Math.round(targetWidth),
        height: Math.round(targetHeight)
    };
}

// --- Routes ---
app.use(express.static('public')); // Checks public folder first

app.get('/', (req, res) => {
    // Fallback if user put index.html in root instead of public
    const rootIndex = path.join(__dirname, 'index.html');
    if (fs.existsSync(rootIndex)) {
        res.sendFile(rootIndex);
    } else {
        res.send('<h1>Error: index.html not found</h1><p>Ensure index.html is in the root folder or public folder.</p>');
    }
});

app.post('/api/crop', upload.single('photo'), async (req, res) => {
    try {
        if (!req.file) return res.status(400).json({ error: 'No file uploaded' });

        // Load image to Canvas
        const img = await canvas.loadImage(req.file.buffer);
        
        // Detect Face (Single Shot Multibox Detector)
        const detections = await faceapi.detectAllFaces(img).withFaceLandmarks();

        if (!detections.length) {
            return res.status(422).json({ error: 'No face detected.' });
        }

        // Get largest face
        const bestFace = detections.reduce((prev, curr) => 
            (prev.detection.box.area > curr.detection.box.area) ? prev : curr
        );

        const cropRegion = calculatePassportCrop(bestFace.detection.box, img.width, img.height);

        // Crop with Sharp
        const outputBuffer = await sharp(req.file.buffer)
            .extract(cropRegion)
            .resize(350, 450)
            .jpeg()
            .toBuffer();

        const returnFormat = req.query.format || 'binary';

        if (returnFormat === 'base64') {
            res.json({ 
                image: `data:image/jpeg;base64,${outputBuffer.toString('base64')}` 
            });
        } else {
            res.set('Content-Type', 'image/jpeg');
            res.send(outputBuffer);
        }

    } catch (err) {
        console.error('Processing Error:', err);
        res.status(500).json({ error: 'Server Error', details: err.message });
    }
});

// Start
loadModels().then(() => {
    app.listen(PORT, () => console.log(`🚀 Ready on port ${PORT}`));
});
