const express = require('express');
const multer = require('multer');
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');
const canvas = require('canvas');
const faceapi = require('@vladmandic/face-api');

// --- Configuration ---
const app = express();
const PORT = 3000;
const upload = multer({ storage: multer.memoryStorage() }); // Store in memory for speed

// --- Face API Setup ---
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// Helper to ensure models exist (Basic Bulletproofing)
// In a production app, you might download these from a CDN at runtime or build time
const MODEL_URL = './models';
async function loadModels() {
    console.log('🏗️  Loading Face API models...');
    try {
        // We use the SSD Mobilenet V1 model for higher accuracy on static images
        await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_URL);
        await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_URL);
        console.log('✅ Models loaded successfully');
    } catch (error) {
        console.error('❌ Error loading models. Make sure you have a "./models" folder with the face-api weights!');
        console.error('   Download them from: https://github.com/justadudewhohacks/face-api.js/tree/master/weights');
        console.error('   You need: ssd_mobilenetv1_model-weights_manifest.json (and shard), face_landmark_68_model-weights_manifest.json (and shard)');
        process.exit(1);
    }
}

// --- Cropping Logic ---
function calculatePassportCrop(box, imageWidth, imageHeight) {
    // Standard Passport Logic: Face should be ~70-80% of the photo height
    // We want a vertical aspect ratio (e.g., 35mm x 45mm = 0.77 ratio)
    
    const faceCenterX = box.x + (box.width / 2);
    const faceCenterY = box.y + (box.height / 2);

    // If face height is 75% of total height
    const targetHeight = box.height / 0.75; 
    const targetWidth = targetHeight * 0.77; // Aspect ratio 3.5:4.5

    // Calculate crop coordinates
    let cropX = faceCenterX - (targetWidth / 2);
    let cropY = faceCenterY - (targetHeight / 2); // Center face vertically
    
    // Adjust Y slightly up because passport photos usually have more space below chin than above head
    cropY = cropY - (box.height * 0.1); 

    // Boundary checks (Bullet proofing)
    if (cropX < 0) cropX = 0;
    if (cropY < 0) cropY = 0;
    if (cropX + targetWidth > imageWidth) cropX = imageWidth - targetWidth;
    if (cropY + targetHeight > imageHeight) cropY = imageHeight - targetHeight;

    // Final safety check if image is too small for calculated crop
    return {
        left: Math.max(0, Math.floor(cropX)),
        top: Math.max(0, Math.floor(cropY)),
        width: Math.min(imageWidth, Math.floor(targetWidth)),
        height: Math.min(imageHeight, Math.floor(targetHeight))
    };
}

// --- Routes ---

app.use(express.static('public'));

app.post('/api/crop', upload.single('photo'), async (req, res) => {
    try {
        if (!req.file) return res.status(400).send('No file uploaded.');

        // 1. Load image into Canvas for FaceAPI
        const img = await canvas.loadImage(req.file.buffer);
        
        // 2. Detect Faces
        // Using SSD Mobilenet for better accuracy than TinyFaceDetector
        const detections = await faceapi.detectAllFaces(img).withFaceLandmarks();

        if (!detections || detections.length === 0) {
            return res.status(422).json({ error: 'No face detected in the photo.' });
        }

        // 3. Select the "Main" Face (Largest Box)
        const bestFace = detections.reduce((prev, current) => {
            return (prev.detection.box.area > current.detection.box.area) ? prev : current;
        });

        const box = bestFace.detection.box;

        // 4. Calculate Crop Region
        const cropRegion = calculatePassportCrop(box, img.width, img.height);

        // 5. Crop using Sharp (Faster and better quality output)
        const outputBuffer = await sharp(req.file.buffer)
            .extract(cropRegion)
            .resize(350, 450) // Resize to standard resolution (optional)
            .toFormat('jpeg')
            .toBuffer();

        // 6. Return based on request type
        const returnFormat = req.query.format || 'binary';

        if (returnFormat === 'base64') {
            const b64 = outputBuffer.toString('base64');
            return res.json({ 
                image: `data:image/jpeg;base64,${b64}`,
                meta: { originalSize: req.file.size, faceConfidence: box.score }
            });
        } else {
            // Default: Binary stream (Displayable in browser directly)
            res.set('Content-Type', 'image/jpeg');
            return res.send(outputBuffer);
        }

    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Processing failed', details: err.message });
    }
});

// Serve the UI
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Start Server
loadModels().then(() => {
    app.listen(PORT, () => {
        console.log(`🚀 Server running on http://localhost:${PORT}`);
    });
});
