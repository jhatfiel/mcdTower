const { cv, cvTranslateError } = require('opencv-wasm');
const fs = require('fs');

const {Jimp, ResizeStrategy} = require('jimp');
const {glob, globSync} = require('glob');
const { createWorker } = require('tesseract.js');
const Tesseract = require('tesseract.js');

const OUT = fs.createWriteStream('_tower.txt');

const HSV_MAT_TYPE = 16;
const RGB_MAT_TYPE = 24;
const E_SIZE = 95;
const E_SIZE_HALF = Math.trunc(E_SIZE/2);
const E_LOC_ARR = [
[1240,708],
[1284,750],
[1328,708],
[1425,708],
[1469,750],
[1513,708],
[1610,708],
[1654,750],
[1698,708],
];
const DEBUG = process.argv.slice(2).includes('--debug');

interface Enchantment {
    fn: string
    name: string
    maxMSE?: number
    offsetX?: number
    offsetY?: number
    meleeName?: string
    rangedName?: string
    armorName?: string
    image?: any
    mask?: any
    bMask?: any
};

interface Item {
    name: string
    possibleNames: Map<string, number>
    enchantments: string[]
    confidence: number[]
    seen: Map<string, number>
    settled?: string
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
            {name: 'Sword', possibleNames: new Map<string, number>([['Sword', 1]]), enchantments: emptyEnchantments(), confidence: emptyConfidence(), seen: new Map<string, number>()},
            {name: 'Mercenary Armor', possibleNames: new Map<string, number>([['Mercenary Armor', 1]]), enchantments: emptyEnchantments(), confidence: emptyConfidence(), seen: new Map<string, number>()},
            {name: 'Bow', possibleNames: new Map<string, number>([['Bow', 1]]), enchantments: emptyEnchantments(), confidence: emptyConfidence(), seen: new Map<string, number>()}
        ]}
    ]
};

function logMatInfo(mat, prefix='') {
    console.error(`${prefix} (RxC)=(${mat.cols}x${mat.rows}), size=(${mat.size().width}*${mat.size().height}), depth=${mat.depth()}, channels=${mat.channels()}, type=${mat.type()}`);
}

function writeGrayscaleImage(img, fn: string) {
    if (!DEBUG) return;
    const imgRGBA = new Uint8Array(img.rows * img.cols * 4);

    for (let i = 0; i < img.rows * img.cols; i++) {
        const value = img.data[i]; // Grayscale value from the img
        imgRGBA[i * 4] = value;    // Red channel
        imgRGBA[i * 4 + 1] = value; // Green channel
        imgRGBA[i * 4 + 2] = value; // Blue channel
        imgRGBA[i * 4 + 3] = 255;   // Alpha channel (fully opaque)
    }
    console.log(`Writing to ${fn}`);
    new Jimp({width: img.cols, height: img.rows, data: Buffer.from(imgRGBA)}).write(fn);
}

function writeRGBAImage(img, fn: string) {
    if (!DEBUG) return;
    console.log(`Writing to ${fn}`);
    new Jimp({width: img.cols, height: img.rows, data: Buffer.from(img.data)}).write(fn);
}

function computeMSE(img1, img2) {
    let diff = new cv.Mat();
    cv.absdiff(img1, img2, diff); // Compute absolute difference between images
    diff.convertTo(diff, cv.CV_32F); // Convert to float for precision

    let squared = new cv.Mat();
    cv.multiply(diff, diff, squared); // Square the differences

    let mean = new cv.Mat();
    cv.reduce(squared, mean, 0, cv.REDUCE_AVG); // Compute mean of squared differences
    cv.reduce(mean, mean, 1, cv.REDUCE_AVG); // Compute mean of squared differences

    let mse = mean.data32F.reduce((sum, v)=>sum+=v,0); // Extract MSE value

    // Clean up
    diff.delete();
    squared.delete();
    mean.delete();

    return mse;
}

function extractImageAsGrayscaleJimp(img, x, y, w, h) {
    let gs = img.roi(new cv.Rect(x, y, w, h));
    cv.cvtColor(gs, gs, cv.COLOR_BGRA2GRAY);
    cv.bitwise_not(gs, gs);
    cv.threshold(gs, gs, 128, 255, cv.THRESH_BINARY);
    const rgba = new Uint8Array(gs.rows * gs.cols * 4);
    for (let i=0; i<gs.rows*gs.cols; i++) {
        rgba[i*4] = rgba[i*4 + 1] = rgba[i*4 + 2] = gs.data[i];
        rgba[i*4 + 3] = 255;
    }
    let result = new Jimp({width: gs.cols, height: gs.rows, data: Buffer.from(rgba)});
    gs.delete();
    return result;
}

(async () => {
    let colorRed = new cv.Scalar(255, 0, 0, 255);
    let colorGreen = new cv.Scalar(0, 255, 0, 255);
    const levelWorker = await createWorker('eng');
    const nameWorker = await createWorker('eng');
    await levelWorker.setParameters({
        tessedit_pageseg_mode: Tesseract.PSM.SINGLE_LINE, // Single line
        tessedit_char_whitelist: 'FOLR/0123456789B:',
        //debug_file: '/dev/null',  // Assigning a debug file disables the console output
    });
    await nameWorker.setParameters({
        tessedit_pageseg_mode: Tesseract.PSM.SINGLE_LINE, // Single line
        tessedit_char_whitelist: `ABCDEFGHIJKLMNOPQRSTUVWXYZ' `,
        debug_file: '/dev/null',  // Assigning a debug file disables the console output
    });

    let now = Date.now();
    process.stderr.write(`Loading ${enchantments.length} enchantment images....`);
    for (let e of enchantments) {
        const src = (await Jimp.read(`./images/${e.fn}`)).resize({w: E_SIZE, h: E_SIZE, mode: ResizeStrategy.HERMITE});
        e.image = cv.matFromImageData(src.bitmap);
        e.offsetX = e.offsetX??0;
        e.offsetY = e.offsetY??0;

        // Extract alpha channel as mask
        let rgba = new cv.MatVector();
        cv.split(e.image, rgba);
        e.bMask = new cv.Mat();
        e.mask = new cv.Mat();
        cv.threshold(rgba.get(3), e.bMask, 1, 255, cv.THRESH_BINARY);

        rgba = new cv.MatVector();
        rgba.push_back(e.bMask);
        rgba.push_back(e.bMask);
        rgba.push_back(e.bMask);
        rgba.push_back(new cv.Mat.zeros(e.bMask.rows, e.bMask.cols, cv.CV_8U));

        cv.merge(rgba, e.mask);
    };
    console.log(`done! (${Math.round((Date.now()-now)/1000)}s)`);

    // item selection screen
    const itemSelectionW = 700;
    const itemSelectionH = 200;
    const itemSelectionLow = new cv.Mat(itemSelectionH, itemSelectionW, RGB_MAT_TYPE, [50, 120, 60, 0]); // lower green
    const itemSelectionHigh = new cv.Mat(itemSelectionH, itemSelectionW, RGB_MAT_TYPE, [60, 140, 70, 255]); // upper green
    // item highlight
    const itemHighlightWidth = 550;
    const itemHighlightHeight = 500;
    const contourLow = new cv.Mat(itemHighlightHeight, itemHighlightWidth, HSV_MAT_TYPE, [0, 0, 250, 0]); // lower white
    const contourHigh = new cv.Mat(itemHighlightHeight, itemHighlightWidth, HSV_MAT_TYPE, [200, 20, 255, 0]); // upper white
    const itemNameWidth = 600;
    const itemNameHeight = 155;
    const itemNameLow = new cv.Mat(itemNameHeight, itemNameWidth, HSV_MAT_TYPE, [0, 0, 250, 0]); // lower white
    const itemNameHigh = new cv.Mat(itemNameHeight, itemNameWidth, HSV_MAT_TYPE, [200, 20, 255, 0]); // upper white

    let sinceLastSelection = 0;
    let lastFloorNum = 0;
    let totalLevels = 0;

    for await (let fn of globSync('videos/*.png').sort()) {
        // SKIP
        if (!['014902'].some(frame => fn.startsWith(`videos/out${frame}`))) continue;
        //if (parseInt(fn.substring(fn.indexOf('0'))) < 11000) continue;
        //if (parseInt(fn.substring(fn.indexOf('0'))) < 4752) continue;

        if (DEBUG) console.log(`${fn} ${'-'.repeat(80)}`);
        else process.stdout.write(`${fn} `);

        let debugImageFN = `debug/${fn.replace(/.*[\/\\]/g, '').replace(/\.png/, '')}`;

        // Read the input image
        now = Date.now();
        const jimpSrc = await Jimp.read(fn);
        if (DEBUG) console.log(`TIMING: (${Math.round((Date.now()-now)/1000)}s) Image Loaded into jimp`); now = Date.now();
        const image = cv.matFromImageData(jimpSrc.bitmap);
        let isSelectionScreen = false;

        // first, see if this is an item selection screen
        const itemSelectImage = image.roi(new cv.Rect(1220, 850, itemSelectionW, itemSelectionH));
        let contourArray = new cv.MatVector();
        let hierarchy = new cv.Mat();
        cv.inRange(itemSelectImage, itemSelectionLow, itemSelectionHigh, itemSelectImage);
        cv.findContours(itemSelectImage, contourArray, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        for (let i=0; i<contourArray.size(); i++) {
            const contour = contourArray.get(i);
            const rect = cv.boundingRect(contour);
            if (rect.width > 500 && rect.height > 100) {
                //console.log(`Found item selection screen!`, rect);
                isSelectionScreen = true;
                break;
            }
        }
        itemSelectImage.delete();
        contourArray.delete();
        hierarchy.delete();

        if (isSelectionScreen) {
            const imageOutput = cv.matFromImageData(jimpSrc.bitmap);
            if (DEBUG) console.log(`TIMING: (${Math.round((Date.now()-now)/1000)}s) Got item selection`); now = Date.now();
            sinceLastSelection = 0;

            // convert image to grayscale and store as jimp image for text recognition of level and next level type
            let imageLevelOffsetX = 300;
            let imageLevelOffsetY = 70;
            let imageGSJimp = extractImageAsGrayscaleJimp(image, imageLevelOffsetX, imageLevelOffsetY, 500, 45);
            if (DEBUG) imageGSJimp.write(`${debugImageFN}_gs.png`);

            if (DEBUG) console.log(`TIMING: (${Math.round((Date.now()-now)/1000)}s) start extract text`); now = Date.now();

            // detect level text
            let nextFloorX = 0;
            let nextFloorY = 0;
            let nextFloorWidth = 200;
            let nextFloorHeight = 45;

            // TODO: Can we "restart" the levelWorker if it just gets in a bad state (or every X uses)?  Does it get in a bad state? (no, doesn't seem to be a bad state)
            // TODO: Could we just keep track of the "expected" level and guess at what the correct level should be when the text recognition fails?
            // TODO: Level # detection seems to be the largest remaining issue (by far) for this to give a 95-98% accurate solution
            // Recognize text and get TSV data
            let result = await levelWorker.recognize(await imageGSJimp.getBuffer("image/png"), {
                rectangle: { top: nextFloorY, left: nextFloorX, width: nextFloorWidth, height: nextFloorHeight},
                rotateAuto: false
            }, {
                text: true,
                blocks: true,
                layoutBlocks: true,
                hocr: true,
                tsv: true,
                box: true,
                unlv: true,
                osd: true,
                debug: true
            });
            let {data: {text: original, words: levelWords, hocr: levelHOCR, tsv: levelTSV, box: levelBOX}} = result;

            if (DEBUG) {
                //console.log(JSON.stringify(result, null, 2));
                console.log('tsv:');
                console.log(levelTSV);
                console.log('hocr:');
                console.log(levelHOCR);
                console.log('box:');
                console.log(levelBOX);
            }

            original = original.trim().replace(/[\\n\\r]/, '');
            let text = original.replace(/B/g, '6');
            text = text.replace(/ /g, '').replace(/:.*$/, '').replace(/^F?L?[0O]?[0O]?R?\s*/, '');
            if (DEBUG) console.log(`TIMING: (${Math.round((Date.now()-now)/1000)}s) extract text from nextFloor area [${text}]`); now = Date.now();

            if (DEBUG) cv.rectangle(imageOutput, new cv.Point(imageLevelOffsetX+nextFloorX, imageLevelOffsetY+nextFloorY), new cv.Point(imageLevelOffsetX+nextFloorX+nextFloorWidth, imageLevelOffsetY+nextFloorY+nextFloorHeight), colorRed, 2, cv.LINE_8, 0);
            if (DEBUG) cv.putText(imageOutput, `[${text}] (${original})`, new cv.Point(imageLevelOffsetX+nextFloorX, imageLevelOffsetY+nextFloorY+nextFloorHeight*2), cv.FONT_HERSHEY_SIMPLEX, 1, colorRed, 2, cv.LINE_8, 0);

            //console.log(`Detected text: [${text}]`);
            if (text.at(-3) !== '/') {
                // HACK
                if (text.at(-3) === '7' && text.length >= 4) text = text.substring(0, text.length-3) + '/' + text.substring(text.length-2);
                else text = text.substring(0, text.length-2) + '/' + text.substring(text.length-2);
            }
            if (DEBUG) console.log(`fixed text: ${text}`);
            let match = text.match(/^(\d\d?)\/+(\d\d)$/);

            if (match) {
                now = Date.now();
                let nextFloorNum = match[1];
                if (nextFloorNum > 30) {
                    // HACK
                    // if it ends in 2, maybe that 2 was just supposed to be part of the slash?
                    if (nextFloorNum%10 === 2) nextFloorNum = Math.trunc(nextFloorNum/10);
                }
                let thisFloorNum = nextFloorNum-1;
                if (thisFloorNum < 30) {
                    totalLevels = Math.max(totalLevels, match[2]);
                    let {data: {text}} = await nameWorker.recognize(await imageGSJimp.getBuffer("image/png"), {
                        rectangle: { top: nextFloorY, left: nextFloorX, width: 400, height: nextFloorHeight}
                    });
                    text = text.toUpperCase();

                    let nextLevelType = bestLDArray(text, tower.floors[nextFloorNum]?.type??'', ['BOSS', 'MERCHANT', 'COMBAT'])?.result;
                    let debugFN = `debug/${thisFloorNum}`;
                    if (!DEBUG) process.stdout.write(`${thisFloorNum}/${totalLevels} `);

                    if (tower.floors[nextFloorNum] === undefined) {
                        tower.floors[nextFloorNum] = { num: nextFloorNum, type: nextLevelType, rewards: []};
                    }
                    lastFloorNum = thisFloorNum;

                    let floor = tower.floors[thisFloorNum];
                    if (floor === undefined) {
                        floor = {num: thisFloorNum, type: 'COMBAT', rewards: []};
                        tower.floors[thisFloorNum] = floor;
                    }
                    if (floor.rewards.filter(r => r !== undefined).length === 5 && floor.rewards.every(r => r.settled !== undefined)) {
                        console.log(`ALL ITEMS SETTLED`);
                    } else {
                        //console.log(`Found floor: ${nextFloor-1}`);

                        // now, figure out which item is highlighted
                        // TODO: only do this with the item selection section, not the entire image
                        // Create the contour image
                        let itemHighlightOffsetX = 660;
                        let itemHighlightOffsetY = 150;
                        let itemHighlightROI = image.roi(new cv.Rect(itemHighlightOffsetX, itemHighlightOffsetY, itemHighlightWidth, itemHighlightHeight));
                        let contourArray = new cv.MatVector();
                        let hierarchy = new cv.Mat();
                        cv.cvtColor(itemHighlightROI, itemHighlightROI, cv.COLOR_BGR2HSV);
                        cv.inRange(itemHighlightROI, contourLow, contourHigh, itemHighlightROI);
                        writeGrayscaleImage(itemHighlightROI, `${debugImageFN}_itemSelection.png`);
                        if (DEBUG) console.log(`TIMING: (${Math.round((Date.now()-now)/1000)}s) wrote itemSelection`); now = Date.now();

                        // Find contours
                        cv.findContours(itemHighlightROI, contourArray, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
                        // set itemNum based on where the bounding box above was found
                        let itemNum: number = undefined;

                        for (let i = 0; i < contourArray.size(); i++) {
                            const contour = contourArray.get(i);
                            const rect = cv.boundingRect(contour);
                            if (rect.width > 120 && rect.height > 160) {
                                //console.log(`Contour ${i} bounding box:`, rect);
                                let col = Math.round(Math.abs(rect.x - 678 + itemHighlightOffsetX)/175);
                                let row = Math.round(Math.abs(rect.y - 178 + itemHighlightOffsetY)/210);
                                itemNum = row*3 + col;
                                cv.rectangle(imageOutput, new cv.Point(itemHighlightOffsetX+rect.x-10, itemHighlightOffsetY+rect.y-10), new cv.Point(itemHighlightOffsetX+rect.x-10+rect.width+20, itemHighlightOffsetY+rect.y-10+rect.height+20), colorRed, 2, cv.LINE_8, 0);
                                //cv.rectangle(imageOutput, new cv.Point(rect.x-10, rect.y-10), new cv.Point(rect.x-10+rect.width+20, rect.y-10+rect.height+20), colorRed, 2, cv.LINE_8, 0);
                                break;
                            }
                        }
                        if (DEBUG) console.log(`TIMING: (${Math.round((Date.now()-now)/1000)}s) found highlighted image ${itemNum}`); now = Date.now();

                        if (itemNum != undefined && itemNum < 5 && floor.rewards[itemNum]?.settled) {
                            let item = floor.rewards[itemNum];
                            if (!DEBUG) console.log(`${itemNum} ${item.settled} SETTLED!`);
                            else console.log(`Skipping item - ${item.settled} SETTLED!`);
                            if (item.enchantments[1] && (!item.enchantments[0] || !item.enchantments[2])) console.log(`!!!! 1 set without 0 or 2`);
                            if (item.enchantments[4] && (!item.enchantments[3] || !item.enchantments[5])) console.log(`!!!! 4 set without 3 or 5`);
                            if (item.enchantments[7] && (!item.enchantments[6] || !item.enchantments[8])) console.log(`!!!! 7 set without 6 or 8`);

                            if ((item.enchantments[0] && !item.enchantments[2]) || (!item.enchantments[0] && item.enchantments[2])) console.log(`!!!! Missing 0/2`);
                            if ((item.enchantments[3] && !item.enchantments[5]) || (!item.enchantments[3] && item.enchantments[5])) console.log(`!!!! Missing 3/5`);
                            if ((item.enchantments[6] && !item.enchantments[8]) || (!item.enchantments[6] && item.enchantments[8])) console.log(`!!!! Missing 6/8`);
                        } else if (itemNum != undefined && itemNum < 5) {
                            if (!DEBUG) process.stdout.write(`${itemNum} `);
                            let hsv = new cv.Mat();

                            let itemNameOffsetX = 1200;
                            let itemNameOffsetY = 180;
                            let itemNameROI = image.roi(new cv.Rect(itemNameOffsetX, itemNameOffsetY, itemNameWidth, itemNameHeight));
                            cv.cvtColor(itemNameROI, itemNameROI, cv.COLOR_BGR2HSV);
                            cv.inRange(itemNameROI, itemNameLow, itemNameHigh, itemNameROI);
                            cv.bitwise_not(itemNameROI, itemNameROI);
                            // Convert single-channel contour image to RGBA for Jimp
                            const imageRGBA = new Uint8Array(itemNameROI.rows * itemNameROI.cols * 4);

                            for (let i = 0; i < itemNameROI.rows * itemNameROI.cols; i++) {
                                const value = itemNameROI.data[i]; // Grayscale value from the contour image
                                imageRGBA[i * 4] = value;    // Red channel
                                imageRGBA[i * 4 + 1] = value; // Green channel
                                imageRGBA[i * 4 + 2] = value; // Blue channel
                                imageRGBA[i * 4 + 3] = 255;   // Alpha channel (fully opaque)
                            }
                            const itemNameJimp = new Jimp({width: itemNameROI.cols, height: itemNameROI.rows, data: Buffer.from(imageRGBA)});
                            if (DEBUG) itemNameJimp.write(`${debugImageFN}_itemName.png`);
                            if (DEBUG) console.log(`TIMING: (${Math.round((Date.now()-now)/1000)}s) wrote itemName`); now = Date.now();

                            // find item name
                            let result = await nameWorker.recognize(await itemNameJimp.getBuffer("image/png"), {
                                rectangle: { top: 210-itemNameOffsetY, left: 1230-itemNameOffsetX, width: itemNameWidth, height: 45},
                                rotateAuto: false
                            }, {
                                text: true,
                                blocks: true,
                                layoutBlocks: true,
                                hocr: true,
                                tsv: true,
                                box: true,
                                unlv: true,
                                osd: true,
                                debug: true
                            });
                            // console.log(JSON.stringify(result, null, 2));
                            let {data: {text: tessName}} = result;
                            if (DEBUG) cv.rectangle(imageOutput, new cv.Point(1230, 210), new cv.Point(1230+itemNameWidth, 210+45), colorRed, 2, cv.LINE_8, 0);
                            if (DEBUG) cv.putText(imageOutput, `[${tessName}]`, new cv.Point(30, 900), cv.FONT_HERSHEY_SIMPLEX, 1, colorRed, 2, cv.LINE_8, 0);
                            // see if we need the second line
                            let countWhite = 0;
                            for (let offset=0; offset<30; offset++) {
                                let pixelValue = image.ucharPtr(265, 1280+offset);
                                if (pixelValue[0] > 128) countWhite++;
                            }
                            if (countWhite) {
                                cv.rectangle(imageOutput, new cv.Point(1280, 265), new cv.Point(1280+30,265), colorGreen, 3, cv.LINE_8, 0);
                                let {data: {text: line2}} = await nameWorker.recognize(await itemNameJimp.getBuffer("image/png"), {
                                    rectangle: { top: 260-itemNameOffsetY, left: 1230-itemNameOffsetX, width: itemNameWidth, height: 45}
                                });
                                cv.rectangle(imageOutput, new cv.Point(1230, 260), new cv.Point(1230+itemNameWidth, 260+45), colorRed, 2, cv.LINE_8, 0);
                                cv.putText(imageOutput, `[${line2}]`, new cv.Point(30, 950), cv.FONT_HERSHEY_SIMPLEX, 1, colorRed, 2, cv.LINE_8, 0);
                                tessName += line2;
                            }
                            tessName = tessName.toUpperCase().replace(/\n/g, ' ').trim();

                            // use Damerau-Levenshtein for each item sorted by cosine similarity
                            let {result: itemName, score} = bestLDItem(tessName, floor.rewards[itemNum]?.name??'');
                            if (DEBUG) console.log(`TIMING: (${Math.round((Date.now()-now)/1000)}s) get item name`); now = Date.now();
                            if (DEBUG) console.log(`Found item name: ${itemName} (was ${tessName}) score=${score}`);
                            if (!DEBUG) process.stdout.write(`${itemName} `);
                            cv.putText(imageOutput, `[${itemName}] (${score})`, new cv.Point(30, 1000), cv.FONT_HERSHEY_SIMPLEX, 1, colorRed, 2, cv.LINE_8, 0);

                            //console.log(`Looking at FLOOR=${thisFloorNum}, ITEMNUM=${itemNum}`);
                            let item = floor.rewards[itemNum];
                            if (item === undefined) {
                                item = {name: itemName, possibleNames: new Map<string, number>(), enchantments: emptyEnchantments(), confidence: emptyConfidence(), seen: new Map<string, number>()};
                                floor.rewards[itemNum] = item;
                            }
                            item.possibleNames.set(itemName, (item.possibleNames.get(itemName)??0)+1);
                            let mostFound = 0;
                            let iter = item.possibleNames.entries();
                            while (true) {
                                let entry = iter.next();
                                if (entry.done) break;
                                //console.log(`name=${entry.value[0]} count=${entry.value[1]}`);
                                if (entry.value[1] > mostFound) {
                                    mostFound = entry.value[1];
                                    itemName = entry.value[0];
                                }
                            }
                            item.name = itemName;
                            if (DEBUG) console.log(`TIMING: (${Math.round((Date.now()-now)/1000)}s) get final item name`); now = Date.now();

                            // Check directly at locations and do a short circuiting mean-squared-error for each enchantment image at that particular location
                            if (DEBUG) console.log(`TIMING: (${Math.round((Date.now()-now)/1000)}s) Prepared to scan for enchantments`); now = Date.now();
                            let enchantmentsFound: string[] = [];
                            for (let slot=0; slot<E_LOC_ARR.length; slot++) {
                                let [x,y] = E_LOC_ARR[slot];
                                const imgCache = new Map<string, any>();
                                enchantments.forEach(e => {
                                    let maxMSE = e.maxMSE??3000; // maximum score for match
                                    let key = `${e.offsetX},${e.offsetY}`;
                                    let eImg = imgCache.get(key);
                                    if (eImg === undefined) {
                                        eImg = image.roi(new cv.Rect(x+e.offsetX, y+e.offsetY, E_SIZE, E_SIZE));
                                        imgCache.set(key, eImg);
                                    }
                                    let maskedImg = new cv.Mat();

                                    try {
                                    cv.bitwise_and(eImg, eImg, maskedImg, e.bMask);
                                    //console.log(`${slot}: ${e.name}`);
                                    let score = computeMSE(maskedImg, e.image);
                                    maskedImg.delete();
                                    if (score > maxMSE) return;
                                    enchantmentsFound[slot] = e.name;

                                    if (item.enchantments[slot] !== '' && item.enchantments[slot] !== e.name) {
                                        if (DEBUG) console.log(`!!!!!!!!${fn} Conflict on ${slot} ${e.name} (${score}) - was ${item.enchantments[slot]} (${item.confidence[slot]})`);
                                        enchantmentsFound[slot] = '.@.@.@.@.CONFLICT.@.@.@.@.';
                                    }
                                    if (item.confidence[slot] === 0 || score < item.confidence[slot]) {
                                        item.confidence[slot] = score;
                                        item.enchantments[slot] = e.name;
                                    }

                                    if (DEBUG) {
                                        // highlight the enchantment icon
                                        const points = [];
                                        points[0] = new cv.Point(x+E_SIZE_HALF, y+7);
                                        points[1] = new cv.Point(x+7, y+E_SIZE_HALF);
                                        points[2] = new cv.Point(x+E_SIZE_HALF, y+E_SIZE-7);
                                        points[3] = new cv.Point(x+E_SIZE-7, y+E_SIZE_HALF);
                                        for (let i=0; i<4; i++) {
                                            cv.line(imageOutput, points[i], points[(i+1)%4], colorRed, 2, cv.LINE_8, 0)
                                        }

                                        // draw on the image
                                        let textImage = new cv.Mat.zeros(imageOutput.rows, imageOutput.cols, imageOutput.type());
                                        let pt = new cv.Point(x, y);
                                        if (slot % 3 === 1) { pt.y += 80; }
                                        else { pt.x += 40; }
                                        cv.putText(textImage, `[${e.name}] (${score.toFixed(3)})`, pt, cv.FONT_HERSHEY_SIMPLEX, 1, colorRed, 2, cv.LINE_8, 0);
                                        let rot = cv.getRotationMatrix2D(pt, (slot%3 === 1)?-45:45, 1.0);
                                        cv.warpAffine(textImage, textImage, rot, new cv.Size(imageOutput.cols, imageOutput.rows));
                                        try {
                                        cv.add(imageOutput, textImage, imageOutput);
                                        } catch (err) { console.trace(cvTranslateError(cv, err)); }
                                        textImage.delete();
                                        rot.delete();

                                        console.log(`SLOT: ${slot}, score=${score}, for ${e.name}`);
                                    }

                                    } catch (err) { console.trace(cvTranslateError(cv, err)); }
                                });
                                [...imgCache.values()].forEach(eImg => eImg.delete());
                                //if (eName) console.log(`------------ Slot: ${slot} (${x},${y}) = ${eName} score=${minMSE}`);
                            }
                            if (DEBUG) console.log(`TIMING: (${Math.round((Date.now()-now)/1000)}s) MSE enchantment icons (${Date.now()-now}ms)`); now = Date.now();

                            let itemSeen = `${itemName} - ${enchantmentsFound.join('/')}`;
                            let seenCount = (item.seen.get(itemSeen)??0) + 1;
                            item.seen.set(itemSeen, seenCount);
                            if (seenCount === 5) {
                                item.settled = itemSeen;
                                item.enchantments = enchantmentsFound;
                            }

                            writeRGBAImage(imageOutput, `${debugImageFN}.png`);
                            writeRGBAImage(imageOutput, `${debugFN}_${itemNum}.png`);

                            if (!DEBUG) console.log(`- ${enchantmentsFound.join('/')}${item.settled?' SETTLED!':''}`);

                            // Clean up
                            hsv.delete();
                            itemNameROI.delete();
                        } else {
                            if (!DEBUG) console.log('No Item Selected');
                        }

                        // Clean up
                        contourArray.delete();
                        hierarchy.delete();
                        itemHighlightROI.delete();
                    }
                } else {
                    console.log(`Invalid floor: [${text}] (original=[${original}])`)
                }
            } else {
                writeRGBAImage(imageOutput, `${debugImageFN}.png`);
                console.log(`!!! UNMATCHED level text: [${text}] (original=[${original}])`);
                //console.log(JSON.stringify(result, null, 2));
            }
            imageOutput.delete();
        } else {
            sinceLastSelection++;
            if (sinceLastSelection > 10) {
                // we could delete these image files once we are correctly finding all item selection screens
                fs.unlinkSync(fn);
                console.log(`!!! UNLINK ${fn}`);
            } else {
                console.log(`Not level`);
            }
            if (sinceLastSelection === 10) {
                if (DEBUG) outputFloorRewards(lastFloorNum);
                OUT.write(getFloorRewards(lastFloorNum) + "\n");
            }
        }

        image.delete();
    }

    itemSelectionLow.delete();
    itemSelectionHigh.delete();
    contourLow.delete();
    contourHigh.delete();

    await levelWorker.terminate();
    await nameWorker.terminate();

    let content = '';
    for (let i=0; i<totalLevels; i++) {
        content += getFloorRewards(i) + '\n';
    }
    fs.writeFileSync('tower.txt', content);
})();

function getFloorRewards(floorNum) {
    let floor = tower.floors[floorNum];
    if (!floor || floor.type === 'MERCHANT') {
        return `${floorNum}\t\t`;
    } else {
        let content = [];
        let numRewards = (floorNum === 0)?3:5;
        for (let i=0; i<numRewards; i++) {
            let reward = floor.rewards[i] ?? { name: 'N/A', possibleNames: new Map<string, number>(), enchantments: emptyEnchantments(), confidence: emptyConfidence()};
            let name = reward.name;
            content.push(`${floorNum}\t${name}\t${reward.enchantments.map(e=>e??'').join('\t')}`);
        }
        return content.join('\n');
    }
}

function outputFloorRewards(floorNum) {
    let content = getFloorRewards(floorNum);
    if (content) console.log(content);
}

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
//'Corrupted Pumpkin',
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

const OCR_TRANSPOSE = {
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
    'LC': 0.2,
    'CL': 0.2,
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
    'HM': 0.4,
    'MH': 0.4,
    'HA': 0.4,
    'AH': 0.4,
    'SC': 0.4,
    'CS': 0.4,
    'LG': 0.4,
    'GL': 0.4,
    'QO': 0.2,
    'OQ': 0.2,
    '[C': 0.2,
    'C[': 0.2,
    'EB': 0.4,
    'BE': 0.4,
    'EW': 0.4,
    'WE': 0.4,
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
            let substitutionCost = (s[i] === t[j])?v0[j]:(v0[j]+(OCR_TRANSPOSE[s[i]+t[j]]??1));
            v1[j+1] = Math.min(deletionCost, insertionCost, substitutionCost);
        }
        /*
        if (i > 0 && v1[n] >= best && v1[n] >= v0[n]) {
            //console.log(`Rejecting ${t} because ${v0[n]} >= ${best} at ${i}`);
            return Infinity;
        }
        */
        v0 = [...v1];
    }

    return v0[n];
}

function bestLDItem(s: string, expected: string): {result: string, score: number} {
    return bestLDArray(s, expected, items);
}

function bestLDArray(s: string, expected: string, arr: string[]): {result: string, score: number} {
    let score = LevenshteinDistance(s, expected.toUpperCase(), Infinity);
    if (expected === '') score = Infinity;
    let result = expected;
    for (let item of arr.filter(item => item !== expected.toUpperCase())) {
        let t = item.toUpperCase();
        let sScore = LevenshteinDistance(s, t, score);
        if (sScore === 0) {
            //console.log(`New best: ${item}, ${sScore} (PERFECT MATCH)`);
            result = item;
            break;
        }
        if (sScore < score) {
            score = sScore;
            result = item;
            //console.log(`New best: ${item}, ${sScore}`);
        }
    }
    return {result, score};
}

const enchantments: Enchantment[] = [
{fn: 'Accelerate.png', name: 'Accelerate'},
{fn: 'Acrobat.png', name: 'Acrobat'},
{fn: 'Ambush.png', name: 'Ambush'},
{fn: 'Anima_Conduit.png', name: 'Anima Conduit', offsetX: -1, offsetY: 0, maxMSE: 4000},
{fn: 'Artifact_Charge_(MCD_Enchantment).png', name: 'Artifact Charge'},
{fn: 'Artifact_Synergy_(MCD_Enchantment).png', name: 'Artifact Synergy'},
{fn: 'Bag_of_Souls.png', name: 'Bag of Souls'},
{fn: 'Beast_Boss_(MCD_Enchantment).png', name: 'Beast Boss'},
{fn: 'Beast_Burst_(MCD_Enchantment).png', name: 'Beast Burst'},
{fn: 'Beast_Surge_(MCD_Enchantment).png', name: 'Beast Surge'},
{fn: 'Bonus_Shot.png', name: 'Bonus Shot'},
{fn: 'Burning.png', name: 'Burning'},
{fn: 'Burst_Bowstring.png', name: 'Burst Bowstring', offsetX: -1, offsetY: 0},
{fn: 'BusyBee.png', name: 'Busy Bee'},
{fn: 'Chain_Reaction.png', name: 'Chain Reaction'},
{fn: 'Chains.png', name: 'Chains'},
{fn: 'Chilling.png', name: 'Chilling', maxMSE: 2500},
{fn: 'Committed.png', name: 'Committed'},
{fn: 'Cool_Down.png', name: 'Cool Down'},
{fn: 'Cooldown_Shot_(MCD_Enchantment).png', name: 'Cooldown Shot'},
{fn: 'Cowardice.png', name: 'Cowardice'},
{fn: 'Critical_Hit.png', name: 'Critical Hit'},
{fn: 'Death_Barter.png', name: 'Death Barter'},
{fn: 'Deflect.png', name: 'Deflect'},
{fn: 'Dipping_Poison.png', name: 'Dipping Poison'},
{fn: 'DynamoMelee.png', name: 'Dynamo'}, // this one is 256x255 (instead of normal 256x256)
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
{fn: 'FireFocus_(MCD_Enchantment).png', name: 'Fire Focus'}, // this one is 256x255 (instead of normal 256x256)
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
{fn: 'Pain_Cycle_(MCD_Enchantment).png', name: 'Pain Cycle', maxMSE: 1000},
{fn: 'Piercing.png', name: 'Piercing'},
{fn: 'Poison_Cloud.png', name: 'Poison Cloud', maxMSE: 1000},
{fn: 'PoisonFocus_(MCD_Enchantment).png', name: 'Poison Focus'},
{fn: 'Potion_Barrier.png', name: 'Potion Barrier', maxMSE: 1500},
{fn: 'Power.png', name: 'Power'},
//{fn: 'Prospector.png', name: 'Prospector'},
{fn: 'Protection.png', name: 'Protection'},
{fn: 'Punch.png', name: 'Punch'},
{fn: 'Radiance.png', name: 'Radiance'},
{fn: 'T_RadianceRanged_Icon.png', name: 'Radiance'},
{fn: 'Rampaging.png', name: 'Rampaging'},
{fn: 'Rapid_Fire.png', name: 'Rapid Fire'},
{fn: 'Reckless_(MCD_Enchantment).png', name: 'Reckless'},
{fn: 'Recycler.png', name: 'Recycler'},
{fn: 'Refreshment_Melee_(MCD_Enchantment).png', name: 'Refreshment', maxMSE: 1800},
{fn: 'Refreshment_Ranged_(MCD_Enchantment).png', name: 'Refreshment', maxMSE: 1800},
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