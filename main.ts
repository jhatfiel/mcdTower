const { cv, cvTranslateError } = require('opencv-wasm');
const fs = require('fs');

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
    fn: string
    name: string
    meleeName?: string
    rangedName?: string
    armorName?: string
    image?: any
    mask?: any
};

interface Item {
    name: string
    possibleNames: Map<string, number>
    enchantments: string[]
    confidence: number[]
};

interface EnchantmentMatch {
    name: string
    score: number
};

interface Floor {
    num: number
    type: string
    rewards: Item[]
}

interface Tower {
    floors: Floor[]
};

function emptyEnchantments(): string[] { return new Array(9).fill(''); }
function emptyConfidence(): number[] { return new Array(9).fill(0); }

const tower: Tower = {
    floors: [
        {num: 0, type: 'START', rewards: [
            {name: 'Sword', possibleNames: new Map<string, number>([['Sword', 1]]), enchantments: emptyEnchantments(), confidence: emptyConfidence()},
            {name: 'Mercenary Armor', possibleNames: new Map<string, number>([['Mercenary Armor', 1]]), enchantments: emptyEnchantments(), confidence: emptyConfidence()},
            {name: 'Bow', possibleNames: new Map<string, number>([['Bow', 1]]), enchantments: emptyEnchantments(), confidence: emptyConfidence()}
        ]}
    ]
};

function logMatInfo(mat, prefix='') {
    console.error(`${prefix} (RxC)=(${mat.cols}x${mat.rows}), size=(${mat.size().width}*${mat.size().height}), depth=${mat.depth()}, channels=${mat.channels()}, type=${mat.type()}`);
}

(async () => {
    //try {
    const worker = await createWorker('eng');
    /*
    await worker.setParameters({
        tessedit_pageseg_mode: Tesseract.PSM.SINGLE_LINE, // Single line
        tessedit_char_whitelist: '/0123456789: COMBATMERCHANTBOSS',
    });
    */
    await worker.setParameters({
        tessedit_pageseg_mode: Tesseract.PSM.SINGLE_LINE, // Single line
        debug_file: '/dev/null',  // Assigning a debug file disables the console output
    });
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

    let now = Date.now();
    process.stderr.write(`Loading ${enchantments.length} enchantment images....`);
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
    console.log(`done! (${Math.round((Date.now()-now)/1000)}s)`);

    // item selection screen
    //const itemSelectionL = new cv.Mat(HEIGHT, WIDTH, HSV_MAT_TYPE, [120, 50, 50, 0]); // lower green
    //const itemSelectionH = new cv.Mat(HEIGHT, WIDTH, HSV_MAT_TYPE, [140, 70, 60, 0]); // upper green
    const itemSelectionL = new cv.Mat(HEIGHT, WIDTH, RGB_MAT_TYPE, [50, 120, 60, 0]); // lower green
    const itemSelectionH = new cv.Mat(HEIGHT, WIDTH, RGB_MAT_TYPE, [60, 140, 70, 255]); // upper green
    // item highlight
    const itemHighlightL = new cv.Mat(HEIGHT, WIDTH, HSV_MAT_TYPE, [0, 0, 250, 0]); // lower white
    const itemHighlightH = new cv.Mat(HEIGHT, WIDTH, HSV_MAT_TYPE, [200, 20, 255, 0]); // upper white
    let sinceLastSelection = 0;
    let lastFloorNum = 0;
    let sampleLocations = [
        [1245, 888],
        [1285, 888],
        [1285, 928],
        [1245, 928],
    ]

    //for await (let fn of globSync('videos/out000550.png').sort()) {
    for await (let fn of globSync('videos/*.png').sort()) {
    //for await (let fn of globSync('test/*.png').sort()) {
    //for await (let fn of ['out000547.png']) {
        //if (!fn.startsWith('test/out000557')) continue;
        //if (!fn.startsWith('test/out000567')) continue; // nautical crossbow
        //if (!fn.startsWith('videos\\out00059')) continue;
        //if (!fn.startsWith('videos\\out000615')) continue;
        console.log(fn);
        let found = false;

        // Read the input i/mage
        const jimpSrc = await Jimp.read(fn);
        const image = cv.matFromImageData(jimpSrc.bitmap);
        let isSelectionScreen = true;
        for (let [x,y] of sampleLocations) {
            let pixelValue = image.ucharPtr(y, x);
            if (pixelValue[0] < 45 || pixelValue[0] > 65 ||
                pixelValue[1] < 120 || pixelValue[1] > 140 ||
                pixelValue[2] < 60 || pixelValue[2] > 80) {
                isSelectionScreen = false; // if the buttons don't match the basic green color, this can't be an item selection screen
                break;
            }
        }

        if (isSelectionScreen) {
            //logMatInfo(image, 'Base image');
            const itemSelectImage = new cv.Mat();

            // first, see if this is an item selection screen
            let contours = new cv.MatVector();
            let hierarchy = new cv.Mat();
            cv.inRange(image, itemSelectionL, itemSelectionH, itemSelectImage);
            cv.findContours(itemSelectImage, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

            for (let i=0; i<contours.size(); i++) {
                const contour = contours.get(i);
                const rect = cv.boundingRect(contour);
                if (rect.x > 1200 && rect.width > 500 && rect.height > 100) {
                    //console.log(`Found item selection screen!`, rect);
                    isSelectionScreen = true;
                    break;
                }
            }
            itemSelectImage.delete();
            contours.delete();
            hierarchy.delete();
        }

        if (isSelectionScreen) {
            sinceLastSelection = 0;
            // crop from contour image?
            let nextFloorX = 385;
            let nextFloorY = 75;
            let nextFloorWidth = 400;
            let nextFloorHeight = 35;

            let imageGS = new cv.Mat();
            cv.cvtColor(image, imageGS, cv.COLOR_BGRA2GRAY);
            cv.threshold(imageGS, imageGS, 128, 255, cv.THRESH_BINARY);
            const imageRGBA = new Uint8Array(imageGS.rows * imageGS.cols * 4);
            for (let i=0; i<imageGS.rows*imageGS.cols; i++) {
                const value = imageGS.data[i];
                imageRGBA[i*4] = value;
                imageRGBA[i*4 + 1] = value;
                imageRGBA[i*4 + 2] = value;
                imageRGBA[i*4 + 3] = 255;
            }
            let imageGSJimp = new Jimp({width: imageGS.cols, height: imageGS.rows, data: Buffer.from(imageRGBA)});

            //let nextFloor = image.roi({x: nextFloorX, y: nextFloorY, width: nextFloorWidth, height: nextFloorHeight});
            //let matchResult = new cv.Mat();
            //cv.matchTemplate(slashImage, nextFloor, matchResult, cv.TM_SQDIFF_NORMED);
            //const widthToSlash = cv.minMaxLoc(matchResult).minLoc.x;
            // resize it to end at the slash
            //nextFloor = image.roi({x: nextFloorX, y: nextFloorY, width: widthToSlash, height: nextFloorHeight});
            //const nextFloorGS = new cv.Mat();
            //cv.cvtColor(nextFloor, nextFloorGS, cv.COLOR_BGR2GRAY);
            //cv.threshold(nextFloorGS, nextFloorGS, 128, 255, cv.THRESH_BINARY);

            // or just use original jimp image?
            /*
            const nextFloorJimp = jimpSrc.crop({x: nextFloorX, y: nextFloorY, w: nextFloorWidth, h: nextFloorHeight});
            /*/
            //*
            /*
            const nextFloorRGBA = new Uint8Array(nextFloorGS.rows * nextFloorGS.cols * 4);

            for (let i = 0; i < nextFloorGS.rows * nextFloorGS.cols; i++) {
                const value = nextFloorGS.data[i]; // Grayscale value from the mask
                nextFloorRGBA[i * 4] = value;    // Red channel
                nextFloorRGBA[i * 4 + 1] = value; // Green channel
                nextFloorRGBA[i * 4 + 2] = value; // Blue channel
                nextFloorRGBA[i * 4 + 3] = 255;   // Alpha channel (fully opaque)
            }

            let nextFloorJimp = new Jimp({width: nextFloorGS.cols, height: nextFloorGS.rows, data: Buffer.from(nextFloorRGBA)})
            */
            //let nextFloorJimp = jimpSrc.crop({x: nextFloorX, y: nextFloorY, w: nextFloorWidth, h: nextFloorHeight});
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
        
            //const pngBuffer = await nextFloorJimp.getBuffer("image/png");

            /*
            let {data: {text}} = await worker.recognize(pngBuffer).then(({ data: {words}}) => {
                const wordConfidences = words.map(word => word.confidence);
                console.log('Word Confidences:', wordConfidences);
            });
            */
            let {data: {text}} = await worker.recognize(await imageGSJimp.getBuffer("image/png"), {
                rectangle: { top: nextFloorY, left: nextFloorX, width: nextFloorWidth, height: nextFloorHeight}
            });
            text = text.replace('\n', ' ');
            //console.log(`Detected text: [${text}]`);
            let match = text.match(/^\s*(\d+)\s*\/\s*(\d+)\s*\:\s*(.*)$/);
            if (match) {
                now = Date.now();
                let nextFloorNum = match[1];
                let thisFloorNum = nextFloorNum-1;
                let numLevels = match[2];
                let nextLevelType = match[3];

                if (tower.floors[nextFloorNum] === undefined) {
                    tower.floors[nextFloorNum] = { num: nextFloorNum, type: nextLevelType, rewards: []};
                }
                lastFloorNum = thisFloorNum;

                let floor = tower.floors[thisFloorNum];
                if (floor === undefined) {
                    floor = {num: thisFloorNum, type: 'COMBAT', rewards: []};
                    tower.floors[thisFloorNum] = floor;
                }

                //console.log(`Found floor: ${nextFloor-1}`);
                //found = true;

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
                const imageRGBA = new Uint8Array(mask.rows * mask.cols * 4);

                for (let i = 0; i < mask.rows * mask.cols; i++) {
                    const value = mask.data[i]; // Grayscale value from the mask
                    imageRGBA[i * 4] = value;    // Red channel
                    imageRGBA[i * 4 + 1] = value; // Green channel
                    imageRGBA[i * 4 + 2] = value; // Blue channel
                    imageRGBA[i * 4 + 3] = 255;   // Alpha channel (fully opaque)
                }
                const imageRGBAJimp = new Jimp({width: mask.cols, height: mask.rows, data: Buffer.from(imageRGBA)});

                //new Jimp({width: mask.cols, height: mask.rows, data: Buffer.from(imageRGBA)}).write('mask.png');
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
                // TODO: set itemNum based on where the bounding box above was found
                let itemNum: number = undefined;

                for (let i = 0; i < contours.size(); i++) {
                    const contour = contours.get(i);
                    const rect = cv.boundingRect(contour);
                    if (rect.width > 120 && rect.height > 160) {
                        //console.log(`Contour ${i} bounding box:`, rect);
                        let col = Math.round(Math.abs(rect.x - 678)/175);
                        let row = Math.round(Math.abs(rect.y - 178)/210);
                        itemNum = row*3 + col;
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

                if (itemNum != undefined && itemNum < 5) {
                    // find item name
                    let {data: {text: line1}} = await worker.recognize(await imageRGBAJimp.getBuffer("image/png"), {
                        rectangle: { top: 210, left: 1230, width: 600, height: 45}
                    });
                    let {data: {text: line2}} = await worker.recognize(await imageRGBAJimp.getBuffer("image/png"), {
                        rectangle: { top: 260, left: 1230, width: 600, height: 45}
                    });
                    let tessName = (`${line1}${line2}`).replace(/\n/g, ' ').trim();
                    // use Damerau-Levenshtein for each item sorted by cosine similarity
                    let {result, score} = bestLD(tessName, floor.rewards[itemNum]?.name??'');
                    console.log(`Found item name: ${result} (was ${tessName}) score=${score}`);

                    //console.log(`Looking at FLOOR=${thisFloorNum}, ITEMNUM=${itemNum}`);
                    let item = floor.rewards[itemNum];
                    if (item === undefined) {
                        item = {name: result, possibleNames: new Map<string, number>(), enchantments: emptyEnchantments(), confidence: emptyConfidence() };
                        floor.rewards[itemNum] = item;
                    }
                    item.possibleNames.set(result, (item.possibleNames.get(result)??0)+1);
                    let mostFound = 0;
                    let iter = item.possibleNames.entries();
                    while (true) {
                        let entry = iter.next();
                        if (entry.done) break;
                        //console.log(`name=${entry.value[0]} count=${entry.value[1]}`);
                        if (entry.value[1] > mostFound) {
                            mostFound = entry.value[1];
                            result = entry.value[0];
                        }
                    }
                    item.name = result;

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
                        //process.stdout.clearLine(1);
                        //process.stdout.write(`${(ind+1).toString().padStart(3)}/${enchantments.length} Searching for enchantment: ${e.name}\r`);
                        let matchResult = new cv.Mat();
                        //console.log(`Trying matchTemplate`);
                        try {
                        cv.matchTemplate(roi, e.image, matchResult, cv.TM_CCORR_NORMED, e.mask);
                        } catch (err) { console.trace(cvTranslateError(cv, err)); }
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
                                    if (row === 1) slot += 1;
                                    else slot += col*2;
                                    //console.log(`FOUND ${e.name} MATCH! ${xReal},${yReal}/${col},${row} (${slot}) = ${score}`);
                                    /*
                                    let pointA = new cv.Point(xReal, yReal);
                                    let pointB = new cv.Point(xReal + e.image.cols, yReal + e.image.rows);
                                    rectArr.push([pointA, pointB]);
                                    */
                                
                                    if (item.enchantments[slot] !== '' && item.enchantments[slot] !== e.name) {
                                        console.log(`!!!!!!!!${fn} Conflict on ${slot} ${e.name} (${score}) - was ${item.enchantments[slot]} (${item.confidence[slot]})`);
                                    }
                                    if (score > item.confidence[slot]) {
                                        item.confidence[slot] = score;
                                        item.enchantments[slot] = e.name;
                                    }
                                }
                            }
                        }
                        //*/
                        
                        // Clean up
                        matchResult.delete();
                    });
                    //let color = new cv.Scalar(0, 255, 0, 255);
                    //rectArr.forEach(r => cv.rectangle(image, r[0], r[1], color, 2, cv.LINE_8, 0));
                    //process.stdout.clearLine(1);
                    //console.log(`Finished scanning for enchantments! (${Math.trunc(Date.now()-now)/1000}s)`);

                    //console.log(`${thisFloorNum} | ${item.name} | ${item.enchantments.join(' | ')}`);

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
                    //new Jimp({width: image.cols, height: image.rows, data: Buffer.from(image.data)}).write('enchantment.png');

                    // Clean up
                    roi.delete();
                }

                // Clean up
                hsv.delete();
                mask.delete();
                contours.delete();
                hierarchy.delete();
            } else {
                console.log(`!!!${fn} UNMATCHED text: [${text}]`);
            }

            // Clean up
            //nextFloorGS.delete();
            //nextFloor.delete();
            imageGS.delete();
        } else {
            sinceLastSelection++;
            if (sinceLastSelection === 10) outputFloorRewards(lastFloorNum);
            if (sinceLastSelection > 10) {
                // we could delete these image files once we are correctly finding all item selection screens
                fs.unlinkSync(fn);
            }
        }

        image.delete();

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

    outputFloorRewards(lastFloorNum);
    for (let i=0; i<=lastFloorNum; i++) {
        outputFloorRewards(i);

    }

})();

function outputFloorRewards(floorNum) {
    let floor = tower.floors[floorNum];
    if (!floor) return;
    if (floor.type === 'MERCHANT') {
        console.log(`${floorNum}\t\t`);
    } else {
        let numRewards = (floorNum === 0)?3:5;
        for (let i=0; i<numRewards; i++) {
            let reward = floor.rewards[i] ?? { name: 'N/A', possibleNames: new Map<string, number>(), enchantments: emptyEnchantments(), confidence: emptyConfidence()};
            let name = reward.name;
            console.log(`${floorNum}\t${name}\t${reward.enchantments.join('\t')}`);
        }
    }

}

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

const items: string[] = [
'Anchor',
'Encrusted Anchor',
'Axe',
'Firebrand',
'Highland Axe',
'Backstabber',
'Swift Striker',
'Battlestaff',
'Battlestaff of Terror',
'Growing Staff',
'Boneclub',
'Bone Cudgel',
'Broken Sawblade',
'Mechanized Sawblade',
'Claymore',
'Broadsword',
'Great Axeblade',
'Heartstealer',
'Coral Blade',
'Sponge Striker',
'Cutlass',
'Dancer\'s Sword',
'Nameless Blade',
'Daggers',
'Fangs of Frost',
'Moon Daggers',
'Sheer Daggers',
'Double Axe',
'Cursed Axe',
'Whirlwind',
'Gauntlets',
'Fighter\'s Bindings',
'Maulers',
'Soul Fists',
'Bow',
'Bonebow',
'Twin Bow',
'Bubble Bow',
'Bubble Burster',
'Burst Crossbow',
'Corrupted Crossbow',
'Soul Hunter Crossbow',
'Cog Crossbow',
'Pride of the Piglins',
'Crossbow',
'Azure Seeker',
'The Slicer',
'Dual Crossbows',
'Baby Crossbows',
'Spellbound Crossbows',
'Exploding Crossbow',
'Firebolt Thrower',
'Imploding Crossbow',
'Harpoon Crossbow',
'Nautical Crossbow',
'Heavy Crossbow',
'Doom Crossbow',
'Slayer Crossbow',
'Hunting Bow',
'Ancient Bow',
'Hunter\'s Promise',
'Master\'s Bow',
'Longbow',
'Guardian Bow',
'Red Snake',
'Power Bow',
'Elite Power Bow',
'Sabrewing',
'Battle Robe',
'Splendid Robe',
'Beenest Armor',
'Beehive Armor',
'Champion\'s Armor',
'Hero\'s Armor',
'Climbing Gear',
'Goat Gear',
'Rugged Climbing Gear',
'Dark Armor',
'Titan\'s Shroud',
'Emerald Gear',
'Gilded Glory',
'Opulent Armor',
'Entertainer\'s Garb',
'The Troubadour',
'Evocation Robe',
'Ember Robe',
'Verdant Robe',
'Ghostly Armor',
'Ghost Kindler',
'Grim Armor',
'Wither Armor',
'Guard\'s Armor',
'Hunter\'s Armor',
'Archer\'s Armor',
'Mercenary Armor',
'Renegade Armor',
'Mystery Armor',
'Blast Fungus',
'Boots of Swiftness',
'Buzzy Nest',
'Corrupted Beacon',
'Corrupted Pumpkin',
'Corrupted Seeds',
'Death Cap Mushroom',
'Enchanted Grass',
'Enchanter\'s Tome',
'Eye of the Guardian',
'Fireworks Arrow',
'Fishing Rod',
'Flaming Quiver',
'Ghost Cloak',
'Golem Kit',
'Gong of Weakening',
'Harpoon Quiver',
'Harvester',
'Ice Wand',
'Iron Hide Amulet',
'Light Feather',
'Lightning Rod',
'Love Medallion',
'Powershaker',
'Satchel of Elixirs',
'Satchel of Elements',
'Satchel of Snacks',
'Scatter Mines',
'Shadow Shifter',
'Shock Powder',
'Soul Healer',
'Soul Lantern',
'Spinblade',
'Tasty Bone',
'Thundering Quiver',
'Torment Quiver',
'Totem of Casting',
'Totem of Regeneration',
'Totem of Shielding',
'Tome of Duplication',
'Updraft Tome',
'Vexing Chant',
'Void Quiver',
'Wind Horn',
'Wonderful Wheat',
'Glaive',
'Grave Bane',
'Venom Glaive',
'Great Hammer',
'Hammer of Gravity',
'Stormlander',
'Katana',
'Dark Katana',
'Master\'s Katana',
'Mace',
'Flail',
'Sun\'s Grace',
'Obsidian Claymore',
'The Starless Night',
'Pickaxe',
'Diamond Pickaxe',
'Rapier',
'Bee Stinger',
'Freezing Foil',
'Sickles',
'Nightmare\'s Bite',
'The Last Laugh',
'Soul Knife',
'Eternal Knife',
'Truthseeker',
'Soul Scythe',
'Frost Scythe',
'Jailor\'s Scythe',
'Spear',
'Fortune Spear',
'Whispering Spear',
'Rapid Crossbow',
'Auto Crossbow',
'Butterfly Crossbow',
'Scatter Crossbow',
'Harp Crossbow',
'Lighting Harb Crossbow',
'Shadow Crossbow',
'Veiled Crossbow',
'Shortbow',
'Love Spell Bow',
'Mechanical Shortbow',
'Purple Storm',
'Snow Bow',
'Winter\'s Touch',
'Soul Bow',
'Bow of Lost Souls',
'Nocturnal Bow',
'Soul Crossbow',
'Feral Soul Crossbow',
'Voidcaller',
'Trickbow',
'The Green Menace',
'The Pink Scoundrel',
'Twisting Vine Bow',
'Weeping Vine Bow',
'Void Bow',
'Call of the Void',
'Wind Bow',
'Burst Gust Bow',
'Echo of the Valley',
'Ocelot Armor',
'Shadow Walker',
'Phantom Armor',
'Frost Bite',
'Piglin Armor',
'Golden Piglin Armor',
'Plate Armor',
'Full Metal Armor',
'Reinforced Mail',
'Stalwart Armor',
'Root Rot Armor',
'Black Spot Armor',
'Scale Mail',
'Highland Armor',
'Shulker Armor',
'Sturdy Shulker Armor',
'Snow Armor',
'Frost Armor',
'Soul Robe',
'Souldancer Robe',
'Spelunker Armor',
'Cave Crawler',
'Sprout Armor',
'Living Vines Armor',
'Squid Armor',
'Glow Squid Armor',
'Teleportation Robes',
'Unstable Robes',
'Thief Armor',
'Spider Armor',
'Sword',
'Diamond Sword',
'Hawkbrand',
'Tempest Knife',
'Chill Gale Knife',
'Resolute Tempest Knife',
'Void Touched Blades',
'The Beginning and The End',
'Whip',
'Vine Whip',
'Turtle Armor',
'Nimble Turtle Armor',
'Wolf Armor',
'Black Wolf Armor',
'Fox Armor',
];

const OCR_LOOKUP = {
    'O0': 0.2,
    '0O': 0.2,
    'I1': 0.2,
    '1I': 0.2,
    'B8': 0.4,
    '8B': 0.4,
    'G6': 0.4,
    '6G': 0.4,
    'S5': 0.4,
    '5S': 0.4,
    'Z2': 0.5,
    '2Z': 0.5,
    'LI': 0.2,
    'IL': 0.2,
    'MN': 0.3,
    'NM': 0.3,
    'CG': 0.5,
    'GC': 0.5,
    'RP': 0.5,
    'PR': 0.5,
    'HN': 0.4,
    'NH': 0.4,
    'HA': 0.4,
    'AH': 0.4,
    'SC': 0.4,
    'CS': 0.4,
    'LG': 0.4,
    'GL': 0.4,
}

function LevenshteinDistance(s: string, t: string, best: number): number {
    let m = s.length;
    let n = t.length;
    let v0: number[] = Array.from({length: n+1}, (_, i) => i);
    let v1: number[] = new Array(n+1);

    for (let i=0; i<m; i++) {
        // calculate v1 (current row distances) from the previous row v0
        // first element of v1 is A[i+1][0]
        //   edit distance is delete (i+1) chars from s to match empty t
        v1[0] = i+1;

        // use formula to fill in the rest of the row
        for (let j=0; j<n; j++) {
            // calculating costs for A[i+1][j+1]
            let deletionCost = v0[j+1] + 1;
            let insertionCost = v1[j] + 1;
            let substitutionCost = (s[i] === t[j])?v0[j]:(v0[j]+(OCR_LOOKUP[s[i]+t[j]]??1));
            v1[j+1] = Math.min(deletionCost, insertionCost, substitutionCost);
        }
        if (i > 0 && v1[n] >= best && v1[n] >= v0[n]) {
            //console.log(`Rejecting ${t} because ${v0[n]} >= ${best} at ${i}`);
            return Infinity;
        }
        v0 = [...v1];
    }

    return v0[n];
}

function bestLD(s: string, expected: string): {result: string, score: number} {
    //console.log(`bestLD(${s})`);
    //expected = 'Nautical Crossbow';
    let score = LevenshteinDistance(s, expected.toUpperCase(), Infinity);
    let result = expected;
    for (let item of items.filter(item => item !== expected.toUpperCase())) {
        let t = item.toUpperCase();
        let sScore = LevenshteinDistance(s, t, score);
        if (sScore === 0) {
            result = item;
            break;
        }
        if (sScore < score) {
            score = sScore;
            result = item;
            console.log(`New best: ${item}, ${sScore}`);
        }
    }
    return {result, score};
}

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
{fn: 'Cowardice.png', name: 'Cowardice'},
{fn: 'Critical_Hit.png', name: 'Critical Hit'},
{fn: 'Death_Barter.png', name: 'Death Barter'},
{fn: 'Deflect.png', name: 'Deflect'},
{fn: 'Dipping_Poison.png', name: 'Dipping Poison'},
{fn: 'DynamoMelee.png', name: 'Dynamo'},
{fn: 'DynamoRanged.png', name: 'Dynamo'},
{fn: 'Echo.png', name: 'Echo'},
{fn: 'Electrified.png', name: 'Electrified'},
//{fn: 'Emerald_Shield.png', name: 'Emerald Shield'},
{fn: 'Enigma_Resonator.png', name: 'Enigma Resonator'},
//{fn: 'Environmental_Protection.png', name: 'Environmental Protection'},
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
//{fn: 'Luck_of_the_Sea_(MCD_Enchantment).png', name: 'Luck of the Sea'},
{fn: 'Lucky_ExplorerIcon.png', name: 'Lucky Explorer'},
{fn: 'Multi_Roll.png', name: 'Multi-Roll'},
{fn: 'Multishot.png', name: 'Multishot'},
{fn: 'Multi-Charge_(MCD_Enchantment).png', name: 'Overcharge'},
{fn: 'Pain_Cycle_(MCD_Enchantment).png', name: 'Pain Cycle'},
{fn: 'Piercing.png', name: 'Piercing'},
{fn: 'Poison_Cloud.png', name: 'Poison Cloud'},
{fn: 'PoisonFocus_(MCD_Enchantment).png', name: 'Poison Focus'},
{fn: 'Potion_Barrier.png', name: 'Potion Barrier'},
{fn: 'Power.png', name: 'Power'},
//{fn: 'Prospector.png', name: 'Prospector'},
{fn: 'Protection.png', name: 'Protection'},
{fn: 'Punch.png', name: 'Punch'},
{fn: 'Radiance.png', name: 'Radiance'},
{fn: 'T_RadianceRanged_Icon.png', name: 'Radiance Shot'},
{fn: 'Rampaging.png', name: 'Rampaging'},
{fn: 'Rapid_Fire.png', name: 'Rapid Fire'},
{fn: 'Reckless_(MCD_Enchantment).png', name: 'Reckless'},
{fn: 'Recycler.png', name: 'Recycler'},
{fn: 'Refreshment_Melee_(MCD_Enchantment).png', name: 'Refreshment'},
{fn: 'Refreshment_Ranged_(MCD_Enchantment).png', name: 'Refreshment'},
{fn: 'Ricochet.png', name: 'Ricochet'},
{fn: 'Roll_Charge.png', name: 'Roll Charge'},
//{fn: 'Rushdown.png', name: 'Rushdown'},
{fn: 'Rush_(MCD_Enchantment).png', name: 'Rush'},
{fn: 'Shadow_Blast.png', name: 'Shadow Blast'},
//{fn: 'Shadow_Shot.png', name: 'Shadow Shot'},
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
//{fn: 'Swarm_Resistance.png', name: 'Swarm Resistance'},
{fn: 'Swiftfooted.png', name: 'Swiftfooted'},
{fn: 'T_Swirling_Icon.png', name: 'Swirling'},
{fn: 'Tempo_Theft.png', name: 'Tempo Theft'},
{fn: 'Thorns.png', name: 'Thorns'},
{fn: 'Thundering.png', name: 'Thundering'},
{fn: 'TumbleBee.png', name: 'Tumblebee'},
{fn: 'Unchanting.png', name: 'Unchanting'},
{fn: 'Void_Shot.png', name: 'Void Strike'},
{fn: 'Void_Strike.png', name: 'Void Strike'},
{fn: 'Wild_Rage.png', name: 'Wild Rage'},
{fn: 'Weakening.png', name: 'Weakening'},
];
//].filter(e => ['Pain Cycle', 'Food Reserves', 'Life Boost', 'Shadow Blast', 'Shadow Surge', 'Cooldown'].indexOf(e.name) !== -1);
//].filter(e => ['Food Reserves', 'Pain Cycle'].indexOf(e.name) !== -1);