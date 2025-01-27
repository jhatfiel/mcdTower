const { cv, cvTranslateError } = require('opencv-wasm');

const {Jimp, ResizeStrategy} = require('jimp');
const {glob, globSync} = require('glob');
const { createWorker } = require('tesseract.js');
const Tesseract = require('tesseract.js');
//import { createWorker } from 'tesseract.js'

const WIDTH = 1920;
const HEIGHT = 1080;
const HSV_MAT_TYPE = 16;
const RGB_MAT_TYPE = 24;

interface Enchantment {
    fn: string;
    name: string;
    meleeName?: string;
    rangedName?: string;
    armorName?: string;
};

(async () => {
    //try {
    const worker = await createWorker('eng');
    await worker.setParameters({
        tessedit_pageseg_mode: Tesseract.PSM.SINGLE_LINE, // Single line
        tessedit_char_whitelist: '/0123456789: COMBATMERCHANTBOSS',
    });
    await worker.setParameters({
    });
    //const slashSrc = await Jimp.read('./images/slash.png');
    //const slashImage = cv.matFromImageData(slashSrc.bitmap);
    const enchantmentSrc = (await Jimp.read('./images/Life_Boost.png')).resize({w:92, h:92, mode: ResizeStrategy.HERMITE});
    const enchantmentImage = cv.matFromImageData(enchantmentSrc.bitmap);
    // Split the channels of the template
    const enchantmentRGBA = new cv.MatVector();
    cv.split(enchantmentImage, enchantmentRGBA);
    const enchantmentAlpha = enchantmentRGBA.get(3);
    let enchantmentBinaryMask = new cv.Mat();
    let enchantmentMask = new cv.Mat(enchantmentBinaryMask.rows, enchantmentBinaryMask.cols, cv.CV_8UC4);
    cv.threshold(enchantmentAlpha, enchantmentBinaryMask, 128, 255, cv.THRESH_BINARY);
    //new Jimp({width: enchantmentMask.cols, height: enchantmentMask.rows, data: Buffer.from(enchantmentMask.data)}).write('enchantmentMask.png');
    const encRGBA = enchantmentMask.data; //new Uint8Array(enchantmentMask.rows * enchantmentMask.cols * 4);
    for (let i = 0; i < enchantmentMask.rows * enchantmentMask.cols; i++) {
        const value = enchantmentMask.data[i]; // Grayscale value from the mask
        encRGBA[i * 4] = value;    // Red channel
        encRGBA[i * 4 + 1] = value; // Green channel
        encRGBA[i * 4 + 2] = value; // Blue channel
        encRGBA[i * 4 + 3] = 255;   // Alpha channel (fully opaque)
    }
    new Jimp({width: enchantmentMask.cols, height: enchantmentMask.rows, data: Buffer.from(encRGBA)}).write('enchantmentMask.png');

    // item selection screen
    //const itemSelectionL = new cv.Mat(HEIGHT, WIDTH, HSV_MAT_TYPE, [120, 50, 50, 0]); // lower green
    //const itemSelectionH = new cv.Mat(HEIGHT, WIDTH, HSV_MAT_TYPE, [140, 70, 60, 0]); // upper green
    const itemSelectionL = new cv.Mat(HEIGHT, WIDTH, RGB_MAT_TYPE, [50, 120, 60, 0]); // lower green
    const itemSelectionH = new cv.Mat(HEIGHT, WIDTH, RGB_MAT_TYPE, [60, 140, 70, 255]); // upper green
    // item highlight
    const itemHighlightL = new cv.Mat(HEIGHT, WIDTH, HSV_MAT_TYPE, [0, 0, 250, 0]); // lower white
    const itemHighlightH = new cv.Mat(HEIGHT, WIDTH, HSV_MAT_TYPE, [200, 20, 255, 0]); // upper white

    //for (let fn of globSync('videos/*.png').sort()
    //for await (let fn of globSync('videos/*.png').sort()) {
    for await (let fn of ['out000547.png']) {
        //if (!fn.startsWith('videos\\out00054')) continue;
        console.error(fn);
        let found = false;

        // Read the input i/mage
        const jimpSrc = await Jimp.read(fn);
        const image = cv.matFromImageData(jimpSrc.bitmap);
        const mask = new cv.Mat();

        // first, see if this is an item selection screen
        let contours = new cv.MatVector();
        let hierarchy = new cv.Mat();
        cv.inRange(image, itemSelectionL, itemSelectionH, mask);
        cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        let isSelectionScreen = false;
        for (let i=0; i<contours.size(); i++) {
            const contour = contours.get(i);
            const rect = cv.boundingRect(contour);
            if (rect.x > 1200 && rect.width > 500 && rect.height > 100) {
                console.log(`Found item selection screen!`, rect);
                isSelectionScreen = true;
                break;
            }
        }

        if (isSelectionScreen) {
            found = true;

            let matchResult = new cv.Mat();
            cv.matchTemplate(image, enchantmentImage, matchResult, cv.TM_CCOEFF_NORMED, enchantmentMask);
            for (let y=0; y<matchResult.rows; y++) {
                for (let x=0; x<matchResult.cols; x++) {
                    const score = matchResult.floatAt(y, x);
                    if (score > 0.6) {
                        console.log(`FOUND ENCHANTMENT MATCH! ${x},${y} = ${score}`);
                        let color = new cv.Scalar(0, 255, 0, 255);
                        let pointA = new cv.Point(x, y);
                        let pointB = new cv.Point(x + enchantmentImage.cols, y + enchantmentImage.rows);
                        cv.rectangle(image, pointA, pointB, color, 2, cv.LINE_8, 0);
                    }
                }
            }
            /*
            cv.threshold(matchResult, matchResult, 0.60, 1, cv.THRESH_BINARY);
            matchResult.convertTo(matchResult, cv.CV_8UC1);
            let contours2 = new cv.MatVector();
            let hierarchy2 = new cv.Mat();
    
            cv.findContours(matchResult, contours2, hierarchy2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
            for (let i = 0; i < contours2.size(); ++i) {
                let countour = contours2.get(i).data32S;
                let x = countour[0];
                let y = countour[1];
                
                let color = new cv.Scalar(0, 255, 0, 255);
                let pointA = new cv.Point(x, y);
                let pointB = new cv.Point(x + enchantmentImage.cols, y + enchantmentImage.rows);
                cv.rectangle(image, pointA, pointB, color, 2, cv.LINE_8, 0);
                console.log(`FOUND ENCHANTMENT MATCH! ${x},${y}`);
            }
            */

            /*
            const minMaxLoc = cv.minMaxLoc(matchResult);
            console.log(JSON.stringify(minMaxLoc));
            cv.rectangle(image, new cv.Point(minMaxLoc.maxLoc.x, minMaxLoc.maxLoc.y), new cv.Point(minMaxLoc.maxLoc.x+foodReservesImage.width, minMaxLoc.maxLoc.y+foodReservesImage.height), new cv.Scalar(0,0,255,255),2);
            */
            new Jimp({width: image.cols, height: image.rows, data: Buffer.from(image.data)}).write('enchantment.png');

            // crop from contour image?
            let nextFloor = image.roi({x: 385, y: 75, width: 400, height: 35});
            //let matchResult = new cv.Mat();
            //cv.matchTemplate(slashImage, nextFloor, matchResult, cv.TM_SQDIFF_NORMED);
            //const widthToSlash = cv.minMaxLoc(matchResult).minLoc.x;
            // resize it to end at the slash
            //nextFloor = image.roi({x: 385, y: 75, width: widthToSlash, height: 35});
            const nextFloorGS = new cv.Mat();
            cv.cvtColor(nextFloor, nextFloorGS, cv.COLOR_BGR2GRAY);

            // or just use original jimp image?
            /*
            const nextFloorJimp = jimpSrc.crop({x: 220, y: 75, w: 500, h: 35});
            /*/
            const nextFloorRGBA = new Uint8Array(nextFloorGS.rows * nextFloorGS.cols * 4);

            for (let i = 0; i < nextFloorGS.rows * nextFloorGS.cols; i++) {
                const value = nextFloorGS.data[i]; // Grayscale value from the mask
                nextFloorRGBA[i * 4] = value;    // Red channel
                nextFloorRGBA[i * 4 + 1] = value; // Green channel
                nextFloorRGBA[i * 4 + 2] = value; // Blue channel
                nextFloorRGBA[i * 4 + 3] = 255;   // Alpha channel (fully opaque)
            }

            const nextFloorJimp = new Jimp({width: nextFloorGS.cols, height: nextFloorGS.rows, data: Buffer.from(nextFloorRGBA)})
            nextFloorJimp.write('level.png');
            /*
            const nextFloorJimp = new Jimp({
                width: 500,
                height: 35,
                data: Buffer.from(nextFloorGS.data),
            });
            */
            //nextFloorJimp.write('level.png');
            // */
        
            const pngBuffer = await nextFloorJimp.getBuffer("image/png");

            const {data: {text}} = await worker.recognize(pngBuffer);
            console.log(`Detected text: ${text}`);

            // now, figure out which item is highlighted
            const hsv = new cv.Mat();
            const mask = new cv.Mat();
            const contours = new cv.MatVector();
            const hierarchy = new cv.Mat();

            // Create the mask
            cv.cvtColor(image, hsv, cv.COLOR_BGR2HSV);
            cv.inRange(hsv, itemHighlightL, itemHighlightH, mask);
            // Convert single-channel mask to RGBA for Jimp
            //*
            const rgbaMask = new Uint8Array(mask.rows * mask.cols * 4);

            for (let i = 0; i < mask.rows * mask.cols; i++) {
                const value = mask.data[i]; // Grayscale value from the mask
                rgbaMask[i * 4] = value;    // Red channel
                rgbaMask[i * 4 + 1] = value; // Green channel
                rgbaMask[i * 4 + 2] = value; // Blue channel
                rgbaMask[i * 4 + 3] = 255;   // Alpha channel (fully opaque)
            }

            new Jimp({width: mask.cols, height: mask.rows, data: Buffer.from(rgbaMask)}).write('mask.png');
            //*/

            // Find contours
            cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
            /*
            contours = contours.filter(c => cv.contourArea(contour) > 10000)
            for (let c of contours) {
                contoursFilt
            }
            cv.drawContours(image, contours, -1, [0, 255, 0], 2);
            */

            // Process contours
            for (let i = 0; i < contours.size(); i++) {
                const contour = contours.get(i);
                const rect = cv.boundingRect(contour);
                if (rect.width > 90 && rect.height > 120) {
                    console.log(`Contour ${i} bounding box:`, rect);
                    /*
                    cv.drawContours(image, contours, i, [255, 0, 0, 255], 4);
                    new Jimp({width: image.cols, height: image.rows, data: Buffer.from(image.data)}).write('contours.png');
                    */
                }

                // Optionally, crop and process the region for OCR
                const cropped = image.roi(rect);
                const croppedImage = new cv.Mat();
                cv.cvtColor(cropped, croppedImage, cv.COLOR_BGR2GRAY);
                /*
                const buffer = Buffer.from(cv.imencode('.png', croppedImage).buffer);
                tesseract.recognize(buffer).then(({ data: { text } }) => {
                    console.log(`Detected text: ${text}`);
                });
                */

                cropped.delete();
                croppedImage.delete();
            }

            // Clean up
            hsv.delete();
            mask.delete();
            contours.delete();
            hierarchy.delete();
        }

        image.delete();
        mask.delete();
        contours.delete();
        hierarchy.delete();

        if (found) break;
    };

    itemSelectionL.delete();
    itemSelectionH.delete();
    itemHighlightL.delete();
    itemHighlightH.delete();

    await worker.terminate();
    //} catch (err) {
        //console.trace(cvTranslateError(cv, err));
    //}
})();

/*

const CV = require('./opencv.js');
const JIMP = require('jimp');
const Jimp = JIMP.Jimp;
const tesseract = require('tesseract.js');
(async () => {
    const cv = await CV;
    console.log(cv.getBuildInformation());
    var jimpSrc = await Jimp.read('frame_0001.png');
    var image = cv.matFromImageData(jimpSrc.bitmap);
    let dst = new cv.Mat();

    const hsv = cv.cvtColor(image, dst, cv.COLOR_BGR2HSV);
    const lowerBound = new cv.Mat();
    const upperBound = new cv.Mat();

    const mask = hsv.inRange(lowerBound, upperBound);
    const contours = mask.findContours(cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    contours.forEach(contour => {
        const rect = contour.boundingRect();
        console.log(rect);
        //const cropped = image.getRegion(rect);
        //extractText(cropped); // OCR
    });
    //let dst = new cv.Mat();
    //let M = cv.Mat.ones(5, 5, cv.CV_8U);

    //const buffer = cv.imencode('.png', croppedImage).toString('base64');
    //tesseract.recognize(Buffer.from(buffer, 'base64'))
        //.then(({ data: { text } }) => console.log(`Detected item: ${text}`));
    //cv.dilate(src, dst, M, anchor, 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
    //new Jimp({width: dst.cols, height: dst.rows, data: Buffer.from(dst.data)}).write('output.png');
    image.delete();
    //dst.delete();
})();
*/

/*
(async () => {
    const resolvedCv = await cv;
    console.log(resolvedCv.getVuildInformation());
  })();

  */
/*
const ffmpeg = require('fluent-ffmpeg');
const cv = require('opencv4nodejs');
const tesseract = require('tesseract.js');
const fs = require('fs');

// 1. Extract frames
function extractFrames(videoPath, outputDir, timestamps) {
    timestamps.forEach((timestamp, index) => {
        ffmpeg(videoPath)
            .setStartTime(timestamp)
            .frames(1)
            .output(`${outputDir}/frame_${index}.png`)
            .on('end', () => console.log(`Extracted frame at ${timestamp}`))
            .run();
    });
}

// 2. Analyze frames for colored borders
function analyzeFrame(framePath) {
    const image = cv.imread(framePath);
    const hsv = image.cvtColor(cv.COLOR_BGR2HSV);
    const lowerBound = new cv.Vec3(0, 100, 100); // Example for red borders
    const upperBound = new cv.Vec3(10, 255, 255);

    const mask = hsv.inRange(lowerBound, upperBound);
    const contours = mask.findContours(cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    contours.forEach(contour => {
        const rect = contour.boundingRect();
        const cropped = image.getRegion(rect);
        extractText(cropped); // OCR
    });
}

// 3. Extract text from items
function extractText(croppedImage) {
    const buffer = cv.imencode('.png', croppedImage).toString('base64');
    tesseract.recognize(Buffer.from(buffer, 'base64'))
        .then(({ data: { text } }) => console.log(`Detected item: ${text}`));
}

// Main program
const videoPath = '../tools/Tower105.mp4';
const outputDir = 'frames';
const timestamps = ['00:01:30', '00:05:00']; // Example timestamps

extractFrames(videoPath, outputDir, timestamps);

// Analyze each frame after extraction
/*
fs.readdirSync(outputDir).forEach(file => {
    analyzeFrame(`${outputDir}/${file}`);
});
*/

const enchantments: Enchantment[] = [
{fn: 'Accelerate.png', name: 'Accelerate'},
{fn: 'Acrobat.png', name: 'Acrobat'},
{fn: 'Ambush.png', name: 'Ambush'},
{fn: 'Anima_Conduit.png', name: 'Anima Conduit'},
{fn: 'Artifact_Charge_(MCD_Enchantment).png', name: 'Artifact Charge'},
{fn: 'Artifact_Synergy_(MCD_Enchantment).png', name: 'Artifact Synergy'},
{fn: 'Bag_of_Souls.png', name: 'Bag of Souls'},
{fn: 'Beast_Boss_(MCD_Enchantment).png', name: 'Beast Boss'},
{fn: 'Beast_Burst_(MCD_Enchantment).png', name: 'Beast Burst'},
{fn: 'Beast_Surge_(MCD_Enchantment).png', name: 'Beast Surge'},
{fn: 'Bonus_Shot.png', name: 'Bonus Shot'},
{fn: 'Burning.png', name: 'Burning'},
{fn: 'Burst_Bowstring.png', name: 'Burst Bowstring'},
{fn: 'BusyBee.png', name: 'Busy Bee'},
{fn: 'Chain_Reaction.png', name: 'Chain Reaction'},
{fn: 'Chains.png', name: 'Chains'},
{fn: 'Chilling.png', name: 'Chilling'},
{fn: 'Committed.png', name: 'Committed'},
{fn: 'Cool_Down.png', name: 'Cool Down'},
{fn: 'Cooldown_Shot_(MCD_Enchantment).png', name: 'Cooldown Shot'},
{fn: 'Cowardice.png', name: ''},
{fn: 'Critical_Hit.png', name: ''},
{fn: 'Death_Barter.png', name: ''},
{fn: 'Deflect.png', name: ''},
{fn: 'Dipping_Poison.png', name: ''},
{fn: 'DynamoMelee.png', name: ''},
{fn: 'DynamoRanged.png', name: ''},
{fn: 'Echo.png', name: ''},
{fn: 'Electrified.png', name: ''},
{fn: 'Emerald_Shield.png', name: ''},
{fn: 'Enigma_Resonator.png', name: ''},
{fn: 'Environmental_Protection.png', name: ''},
{fn: 'Exploding.png', name: ''},
{fn: 'Explorer.png', name: ''},
{fn: 'Final_Shout.png', name: ''},
{fn: 'Fire_Aspect.png', name: ''},
{fn: 'FireFocus_(MCD_Enchantment).png', name: ''},
{fn: 'Fire_Trail.png', name: ''},
{fn: 'Food_Reserves.png', name: ''},
{fn: 'Freezing.png', name: ''},
{fn: 'Frenzied.png', name: ''},
{fn: 'Fuse_Shot.png', name: ''},
{fn: 'Gravity.png', name: ''},
{fn: 'Gravity_Pulse.png', name: ''},
{fn: 'Growing.png', name: ''},
{fn: 'Guarding_Strike_(MCD_Enchantment).png', name: ''},
{fn: 'Health_Synergy.png', name: ''},
{fn: 'IllagersBane.png', name: ''},
{fn: 'Infinity.png', name: ''},
{fn: 'Leeching.png', name: ''},
{fn: 'Levitation_Shot.png', name: ''},
{fn: 'Life_Boost.png', name: ''},
{fn: 'LightningFocus_(MCD_Enchantment).png', name: ''},
{fn: 'Looting.png', name: ''},
{fn: 'Luck_of_the_Sea_(MCD_Enchantment).png', name: ''},
{fn: 'Lucky_ExplorerIcon.png', name: ''},
{fn: 'Multi-Charge_(MCD_Enchantment).png', name: ''},
{fn: 'Multi_Roll.png', name: ''},
{fn: 'Multishot.png', name: ''},
{fn: 'Pain_Cycle_(MCD_Enchantment).png', name: ''},
{fn: 'Piercing.png', name: ''},
{fn: 'Poison_Cloud.png', name: ''},
{fn: 'Potion_Barrier.png', name: ''},
{fn: 'Power.png', name: ''},
{fn: 'Prospector.png', name: ''},
{fn: 'Protection.png', name: ''},
{fn: 'Punch.png', name: ''},
{fn: 'Radiance.png', name: ''},
{fn: 'Rampaging.png', name: ''},
{fn: 'Rapid_Fire.png', name: ''},
{fn: 'Reckless_(MCD_Enchantment).png', name: ''},
{fn: 'Recycler.png', name: ''},
{fn: 'Refreshment_Melee_(MCD_Enchantment).png', name: ''},
{fn: 'Refreshment_Ranged_(MCD_Enchantment).png', name: ''},
{fn: 'Ricochet.png', name: ''},
{fn: 'Roll_Charge.png', name: ''},
{fn: 'Rushdown.png', name: ''},
{fn: 'Rush_(MCD_Enchantment).png', name: ''},
{fn: 'Shadow_Blast.png', name: ''},
{fn: 'Shadow_Shot.png', name: ''},
{fn: 'Shadow_Surge_(MCD_Enchantment).png', name: ''},
{fn: 'Shared_Pain.png', name: ''},
{fn: 'Sharpness.png', name: ''},
{fn: 'Shockwave.png', name: ''},
{fn: 'Shock_Web_(MCD_Enchantment).png', name: ''},
{fn: 'slash.png', name: ''},
{fn: 'Smiting.png', name: ''},
{fn: 'Snowball_(Dungeons).png', name: ''},
{fn: 'SoulFocus_(MCD_Enchantment).png', name: ''},
{fn: 'Soul_Siphon.png', name: ''},
{fn: 'Soul_Speed.png', name: ''},
{fn: 'Speed_Synergy.png', name: ''},
{fn: 'Stunning.png', name: ''},
{fn: 'Supercharge.png', name: ''},
{fn: 'Surprise_Gift.png', name: ''},
{fn: 'Swarm_Resistance.png', name: ''},
{fn: 'Swiftfooted.png', name: ''},
{fn: 'Tempo_Theft.png', name: ''},
{fn: 'Thorns.png', name: ''},
{fn: 'Thundering.png', name: ''},
{fn: 'T_RadianceRanged_Icon.png', name: ''},
{fn: 'T_Swirling_Icon.png', name: ''},
{fn: 'TumbleBee.png', name: ''},
{fn: 'Unchanting.png', name: ''},
{fn: 'Void_Shot.png', name: ''},
{fn: 'Void_Strike.png', name: ''},
{fn: 'Wild_Rage.png', name: ''},
]