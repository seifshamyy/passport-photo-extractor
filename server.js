const express = require('express');
const multer = require('multer');
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');
const canvas = require('canvas');
const faceapi = require('@vladmandic/face-api');

// --- Configuration ---
const app = express();
const PORT = process.env.PORT || 3000;
const upload = multer({ storage: multer.memoryStorage() });

// --- Face API Setup ---
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// UPDATED: Use absolute path for models to avoid "missing directory" errors
const MODEL_URL = path.join(__dirname, 'models');

async function loadModels() {
    console.log(`🏗️  Loading Face API models from: ${MODEL_URL}`);
    
    // DEBUG: List files to ensure they actually exist on the server
    try {
        const files = fs.readdirSync(MODEL_URL);
        console.log('✅ Found files in models folder:', files);
    } catch (err) {
        console.error('❌ CRITICAL ERROR: The "models" folder is missing or empty on the server!');
        console.error('   This usually means you forgot to commit the folder to Git.');
        console.error('   Error details:', err.message);
        process.exit(1); 
    }

    try {
        await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_URL);
        await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_URL);
        console.log('✅ Models loaded successfully');
    } catch (error) {
        console.error('❌ Error loading specific model weights.');
        console.error(error); // Print the actual error for debugging
        process.exit(1);
    }
}

// --- Cropping Logic ---
function calculatePassportCrop(box, imageWidth, imageHeight) {
    const faceCenterX = box.x + (box.width / 2);
    const faceCenterY = box.y + (box.height / 2);

    const targetHeight = box.height / 0.75; 
    const targetWidth = targetHeight * 0.77; 

    let cropX = faceCenterX - (targetWidth / 2);
    let cropY = faceCenterY - (targetHeight / 2); 
    
    cropY = cropY - (box.height * 0.1); 

    if (cropX < 0) cropX = 0;
    if (cropY < 0) cropY = 0;
    if (cropX + targetWidth > imageWidth) cropX = imageWidth - targetWidth;
    if (cropY + targetHeight > imageHeight) cropY = imageHeight - targetHeight;

    return {
        left: Math.max(0, Math.floor(cropX)),
        top: Math.max(0, Math.floor(cropY)),
        width: Math.min(imageWidth, Math.floor(targetWidth)),
        height: Math.min(imageHeight, Math.floor(targetHeight))
    };
}

// --- Routes ---
app.use(express.static('public'));

app.get('/', (req, res) => {
    // Check if index.html is in root or public (handling both cases)
    if (fs.existsSync(path.join(__dirname, 'index.html'))) {
        res.sendFile(path.join(__dirname, 'index.html'));
    } else {
        res.sendFile(path.join(__dirname, 'public', 'index.html'));
    }
});

app.post('/api/crop', upload.single('photo'), async (req, res) => {
    try {
        if (!req.file) return res.status(400).send('No file uploaded.');

        const img = await canvas.loadImage(req.file.buffer);
        const detections = await faceapi.detectAllFaces(img).withFaceLandmarks();

        if (!detections || detections.length === 0) {
            return res.status(422).json({ error: 'No face detected in the photo.' });
        }

        const bestFace = detections.reduce((prev, current) => {
            return (prev.detection.box.area > current.detection.box.area) ? prev : current;
        });

        const box = bestFace.detection.box;
        const cropRegion = calculatePassportCrop(box, img.width, img.height);

        const outputBuffer = await sharp(req.file.buffer)
            .extract(cropRegion)
            .resize(350, 450)
            .toFormat('jpeg')
            .toBuffer();

        const returnFormat = req.query.format || 'binary';

        if (returnFormat === 'base64') {
            const b64 = outputBuffer.toString('base64');
            return res.json({ 
                image: `data:image/jpeg;base64,${b64}`,
                meta: { originalSize: req.file.size, faceConfidence: box.score }
            });
        } else {
            res.set('Content-Type', 'image/jpeg');
            return res.send(outputBuffer);
        }

    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Processing failed', details: err.message });
    }
});

// Start Server
loadModels().then(() => {
    app.listen(PORT, () => {
        console.log(`🚀 Server running on http://localhost:${PORT}`);
    });
});
