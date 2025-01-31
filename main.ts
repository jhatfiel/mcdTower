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
    image?: any;
    mask?: any;
};

interface Item {
    name: string;
    enchantments: string[];
};

interface EnchantmentMatch {
    name: string;
    score: number;
};

interface Tower {
    items: Item[][]; // array for each floor completed - 0 is what you start with, 1 is what you get after 1, etc
};

const tower: Tower = {items: [[
    {name: 'Sword', enchantments: []},
    {name: 'Mercenary Armor', enchantments: []},
    {name: 'Bow', enchantments: []},
]]};

function logMatInfo(mat, prefix='') {
    console.log(`${prefix} (RxC)=(${mat.cols}x${mat.rows}), size=(${mat.size().width}*${mat.size().height}), depth=${mat.depth()}, channels=${mat.channels()}, type=${mat.type()}`);
}

(async () => {
    //try {
    const worker = await createWorker('eng');
    /*
    await worker.setParameters({
        tessedit_pageseg_mode: Tesseract.PSM.SINGLE_LINE, // Single line
        tessedit_char_whitelist: '/0123456789: COMBATMERCHANTBOSS',
    });
    await worker.setParameters({
    });
    */
    //const slashSrc = await Jimp.read('./images/slash.png');
    //const slashImage = cv.matFromImageData(slashSrc.bitmap);
    /*
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
    */

    process.stdout.write(`Loading ${enchantments.length} enchantment images....`);
    for (let e of enchantments) {
        const src = (await Jimp.read(`./images/${e.fn}`)).resize({w: 92, h: 92, mode: ResizeStrategy.HERMITE});
        e.image = cv.matFromImageData(src.bitmap);

        // Extract alpha channel as mask
        let rgba = new cv.MatVector();
        cv.split(e.image, rgba);
        e.mask = new cv.Mat();
        cv.threshold(rgba.get(3), e.mask, 1, 255, cv.THRESH_BINARY);

        rgba = new cv.MatVector();
        //rgba.push_back(e.mask);
        rgba.push_back(e.mask);
        rgba.push_back(e.mask);
        rgba.push_back(e.mask);
        rgba.push_back(new cv.Mat.zeros(e.mask.rows, e.mask.cols, cv.CV_8U));

        cv.merge(rgba, e.mask);

        /*
        const data = new Uint8Array(e.mask.rows*e.mask.cols*4);
        for (let i=0; i<e.mask.rows*e.mask.cols; i++) {
            const value = e.mask.data[i];
            data[i*4] = value;
            data[i*4+1] = value;
            data[i*4+2] = value;
            data[i*4+3] = 255;
        }
        new Jimp({width: e.mask.cols, height: e.mask.rows, data: Buffer.from(data)}).write('enchantmentMask.png');
        */
        // the mask should have a CV_8U or CV_32F depth and the same number of channels as the template image
        //logMatInfo(e.image, 'Enchantment Template');
        //logMatInfo(e.mask, `Enchantment Mask`);
    };
    console.log('done!');

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
        logMatInfo(image, 'Base image');
        const itemSelectImage = new cv.Mat();

        // first, see if this is an item selection screen
        let contours = new cv.MatVector();
        let hierarchy = new cv.Mat();
        cv.inRange(image, itemSelectionL, itemSelectionH, itemSelectImage);
        cv.findContours(itemSelectImage, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

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

            let now = Date.now();
            //process.stdout.write(`  0/${enchantments.length} Scanning enchantment images....\r`);
            const xOffset = 1250;
            const yOffset = 700;
            const buffer = 100;
            const roi = image.roi(new cv.Rect(xOffset-buffer, yOffset-buffer, 520+2*buffer, 135+2*buffer));
            //logMatInfo(roi, 'roi');
            /*
            const roiBGR = new cv.Mat();
            cv.cvtColor(roi, roiBGR, cv.COLOR_BGR2BGRA);
            new Jimp({width: roiBGR.cols, height: roiBGR.rows, data: Buffer.from(roiBGR.data)}).write('roi.png');
            */

            let rectArr: any[][] = []; 
            enchantments.forEach((e, ind) => {
                //process.stdout.write(`${ind.toString().padStart(3)}/${enchantments.length} Scanning enchantment images ${e.name}\r`);
                console.log(e.name);
                let matchResult = new cv.Mat();
                //console.log(`Trying matchTemplate`);
                try {
                cv.matchTemplate(roi, e.image, matchResult, cv.TM_CCORR_NORMED, e.mask);
                } catch (err) { console.trace(cvTranslateError(cv, err)); }
                /*
                console.log(`[${e.name}] Score at 1240,709=${matchResult.floatAt(709-yOffset+buffer, 1240-xOffset+buffer)}`);
                console.log(`[${e.name}] Score at 1241,709=${matchResult.floatAt(709-yOffset+buffer, 1241-xOffset+buffer)}`);
                console.log(`[${e.name}] Score at 1240,710=${matchResult.floatAt(710-yOffset+buffer, 1240-xOffset+buffer)}`);
                console.log(`[${e.name}] Score at 1241,710=${matchResult.floatAt(710-yOffset+buffer, 1241-xOffset+buffer)}`);
                console.log(`[${e.name}] Score at 1341,710=${matchResult.floatAt(710-yOffset+buffer, 1341-xOffset+buffer)}`);
                console.log(`[${e.name}] Score at 1441,710=${matchResult.floatAt(710-yOffset+buffer, 1441-xOffset+buffer)}`);
                console.log(`[${e.name}] Score at 1541,710=${matchResult.floatAt(710-yOffset+buffer, 1541-xOffset+buffer)}`);
                console.log(`[${e.name}] Score at 1641,710=${matchResult.floatAt(710-yOffset+buffer, 1641-xOffset+buffer)}`);
                /*/
                for (let y=0; y<matchResult.rows; y++) {
                    for (let x=0; x<matchResult.cols; x++) {
                        const score = matchResult.floatAt(y, x);
                        if (score > 0.92) {
                            let xReal = xOffset-buffer+x;
                            let yReal = yOffset-buffer+y;
                            let slot = 0;
                            let row = Math.round(Math.abs(yReal-709)/41);
                            let col = xReal-1241;
                            if (col > 120) { col -= 185; slot += 3; }
                            if (col > 120) { col -= 185; slot += 3; }
                            col = Math.round(Math.abs(col)/88);
                            if (row === 1) slot += 2;
                            else slot += col;
                            console.log(`FOUND ${e.name} MATCH! ${xReal},${yReal}/${col},${row} (${slot}) = ${score}`);
                            let pointA = new cv.Point(xReal, yReal);
                            let pointB = new cv.Point(xReal + e.image.cols, yReal + e.image.rows);
                            rectArr.push([pointA, pointB]);
                        }
                    }
                }
                //*/
            });
            let color = new cv.Scalar(0, 255, 0, 255);
            rectArr.forEach(r => cv.rectangle(image, r[0], r[1], color, 2, cv.LINE_8, 0));
            console.log(`\nComplete! (${Math.trunc(Date.now()-now)/1000})s`);

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
            //nextFloorJimp.write('level.png');
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
        itemSelectImage.delete();
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
{fn: 'Cool_Down.png', name: 'Cooldown'},
{fn: 'Cooldown_Shot_(MCD_Enchantment).png', name: 'Cooldown Shot'},
{fn: 'Cowardice.png', name: 'Cowardice'},
{fn: 'Critical_Hit.png', name: 'Critical Hit'},
{fn: 'Death_Barter.png', name: 'Death Barter'},
{fn: 'Deflect.png', name: 'Deflect'},
{fn: 'Dipping_Poison.png', name: 'Dipping Poison'},
{fn: 'DynamoMelee.png', name: 'Dynamo'},
{fn: 'DynamoRanged.png', name: 'Dynamo'},
{fn: 'Echo.png', name: 'Echo'},
{fn: 'Electrified.png', name: 'Electrified'},
{fn: 'Emerald_Shield.png', name: 'Emerald Shield'},
{fn: 'Enigma_Resonator.png', name: 'Enigma Resonator'},
{fn: 'Environmental_Protection.png', name: 'Environmental Protection'},
{fn: 'Exploding.png', name: 'Exploding'},
{fn: 'Explorer.png', name: 'Explorer'},
{fn: 'Final_Shout.png', name: 'Final Shout'},
{fn: 'Fire_Aspect.png', name: 'Fire Aspect'},
{fn: 'FireFocus_(MCD_Enchantment).png', name: 'Fire Focus'},
{fn: 'Fire_Trail.png', name: 'Fire Trail'},
{fn: 'Food_Reserves.png', name: 'Food Reserves'},
{fn: 'Freezing.png', name: 'Freezing'},
{fn: 'Frenzied.png', name: 'Frenzied'},
{fn: 'Fuse_Shot.png', name: 'Fuse Shot'},
{fn: 'Gravity.png', name: 'Gravity'},
{fn: 'Gravity_Pulse.png', name: 'Gravity Pulse'},
{fn: 'Growing.png', name: 'Growing'},
{fn: 'Guarding_Strike_(MCD_Enchantment).png', name: 'Guarding Strike'},
{fn: 'Health_Synergy.png', name: 'Health Synergy'},
{fn: 'IllagersBane.png', name: `Illager's Bane`},
{fn: 'Infinity.png', name: 'Infinity'},
{fn: 'Leeching.png', name: 'Leeching'},
{fn: 'Levitation_Shot.png', name: 'Levitation Shot'},
{fn: 'Life_Boost.png', name: 'Life Boost'},
{fn: 'LightningFocus_(MCD_Enchantment).png', name: 'Lightning Focus'},
{fn: 'Looting.png', name: 'Looting'},
{fn: 'Luck_of_the_Sea_(MCD_Enchantment).png', name: 'Luck of the Sea'},
{fn: 'Lucky_ExplorerIcon.png', name: 'Lucky Explorer'},
{fn: 'Multi-Charge_(MCD_Enchantment).png', name: 'Multi Charge'},
{fn: 'Multi_Roll.png', name: 'Multi Roll'},
{fn: 'Multishot.png', name: 'Multishot'},
{fn: 'Pain_Cycle_(MCD_Enchantment).png', name: 'Pain Cycle'},
{fn: 'Piercing.png', name: 'Piercing'},
{fn: 'Poison_Cloud.png', name: 'Poison Cloud'},
{fn: 'Potion_Barrier.png', name: 'Potion Barrier'},
{fn: 'Power.png', name: 'Power'},
{fn: 'Prospector.png', name: 'Prospector'},
{fn: 'Protection.png', name: 'Protection'},
{fn: 'Punch.png', name: 'Punch'},
{fn: 'Radiance.png', name: 'Radiance'},
{fn: 'Rampaging.png', name: 'Rampaging'},
{fn: 'Rapid_Fire.png', name: 'Rapid Fire'},
{fn: 'Reckless_(MCD_Enchantment).png', name: 'Reckless'},
{fn: 'Recycler.png', name: 'Recycler'},
{fn: 'Refreshment_Melee_(MCD_Enchantment).png', name: 'Refreshment'},
{fn: 'Refreshment_Ranged_(MCD_Enchantment).png', name: 'Refreshment'},
{fn: 'Ricochet.png', name: 'Ricochet'},
{fn: 'Roll_Charge.png', name: 'Roll Charge'},
{fn: 'Rushdown.png', name: 'Rushdown'},
{fn: 'Rush_(MCD_Enchantment).png', name: 'Rush'},
{fn: 'Shadow_Blast.png', name: 'Shadow Blast'},
{fn: 'Shadow_Shot.png', name: 'Shadow Shot'},
{fn: 'Shadow_Surge_(MCD_Enchantment).png', name: 'Shadow Surge'},
//{fn: 'Shared_Pain.png', name: 'Shared Pain'},
{fn: 'Sharpness.png', name: 'Sharpness'},
{fn: 'Shockwave.png', name: 'Shockwave'},
{fn: 'Shock_Web_(MCD_Enchantment).png', name: 'Shock Web'},
{fn: 'Smiting.png', name: 'Smiting'},
{fn: 'Snowball_(Dungeons).png', name: 'Snowball'},
{fn: 'SoulFocus_(MCD_Enchantment).png', name: 'Soul Focus'},
{fn: 'Soul_Siphon.png', name: 'Soul Siphon'},
{fn: 'Soul_Speed.png', name: 'Soul Speed'},
{fn: 'Speed_Synergy.png', name: 'Speed Synergy'},
{fn: 'Stunning.png', name: 'Stunning'},
{fn: 'Supercharge.png', name: 'Supercharge'},
{fn: 'Surprise_Gift.png', name: 'Surprise Gift'},
{fn: 'Swarm_Resistance.png', name: 'Swarm Resistance'},
{fn: 'Swiftfooted.png', name: 'Swiftfooted'},
{fn: 'Tempo_Theft.png', name: 'Tempo Theft'},
{fn: 'Thorns.png', name: 'Thorns'},
{fn: 'Thundering.png', name: 'Thundering'},
{fn: 'T_RadianceRanged_Icon.png', name: 'Radiance'},
{fn: 'T_Swirling_Icon.png', name: 'Swirling'},
{fn: 'TumbleBee.png', name: 'Tumblebee'},
{fn: 'Unchanting.png', name: 'Unchanting'},
{fn: 'Void_Shot.png', name: 'Void Shot'},
{fn: 'Void_Strike.png', name: 'Void Strike'},
{fn: 'Wild_Rage.png', name: 'Wild Rage'},
];
//].filter(e => ['Pain Cycle', 'Food Reserves', 'Life Boost', 'Shadow Blast', 'Shadow Surge', 'Cooldown'].indexOf(e.name) !== -1);
//].filter(e => ['Food Reserves', 'Pain Cycle'].indexOf(e.name) !== -1);