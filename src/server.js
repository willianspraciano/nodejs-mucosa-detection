const fs = require('fs/promises');
const path = require('path');

const automl = require('@tensorflow/tfjs-automl');
const tf = require('@tensorflow/tfjs-node');

const sharp = require('sharp');

const color = require('dominant-color');

const vibrant = require('node-vibrant');

let dominantArray = [];
let vibrantArray = [];
let darkVibrantArray = [];
let lightVibrantArray = [];
let mutedArray = [];
let darkMutedArray = [];
let lightMutedArray = [];

function save(outputFile, object) {
  const outputPath = path.resolve(__dirname, 'output', outputFile);
  fs.writeFile(outputPath, JSON.stringify(object, '', 2), { flag: 'w' });
}

function newArray(color, outputName) {
  let colorNumber = [];

  // Se a cor vier em porcentagem
  if (typeof color[0] === 'string' && color[0].includes('%')) {
    colorNumber = color.map((el, i) => {
      el = Number(el.replace('%', ''));

      // converte a porcentagem de para um número de 0 a 255
      return Math.round((el * 255) / 100);
    });
  } else {
    colorNumber = color;
  }

  let arr = {};
  arr.red = Math.round(Number(colorNumber[0]));
  arr.green = Math.round(Number(colorNumber[1]));
  arr.blue = Math.round(Number(colorNumber[2]));
  arr.file = outputName;
  arr.label = 'T';

  return arr;
}

async function getColor(outputName) {
  const imageCropped = path.resolve(__dirname, 'output', outputName);

  console.log('image: ', imageCropped);
  await color(imageCropped, { format: 'rgb' }, function (err, color) {
    // console.log('erro: ', err);
    // console.log('Color: ', color);
    // console.log('Arr: ', newArray(color, outputName));
    dominantArray.push(newArray(color, outputName));

    save('dominant.json', dominantArray);
  });

  vibrant.from(imageCropped).getPalette((err, palette) => {
    console.log(palette);
    console.log(err);
    vibrantArray.push(newArray(palette.Vibrant._rgb, outputName));
    save('vibrant.json', vibrantArray);

    darkVibrantArray.push(newArray(palette.DarkVibrant._rgb, outputName));
    save('darkVibrant.json', darkVibrantArray);

    lightVibrantArray.push(newArray(palette.LightVibrant._rgb, outputName));
    save('lightVibrant.json', lightVibrantArray);

    mutedArray.push(newArray(palette.Muted._rgb, outputName));
    save('muted.json', mutedArray);

    darkMutedArray.push(newArray(palette.DarkMuted._rgb, outputName));
    save('darkMuted.json', darkMutedArray);

    lightMutedArray.push(newArray(palette.LightMuted._rgb, outputName));
    save('lightMuted.json', lightMutedArray);
  });
}

async function cropImage(image, imageName, width, height, left, top) {
  console.log(imageName);

  const outputName = `cropped_${imageName}`;

  await sharp(image)
    .extract({ width, height, left, top })
    .toFile(path.resolve(__dirname, 'output', outputName))
    .then(() => console.log(`Image ${imageName} cropped and saved`))
    .catch((err) => console.log('Error when cropping image'));

  await getColor(outputName);
}

async function getMucosa(imageFolder, imageName) {
  const modelJSON = 'file://src/model/model.json';
  const imagePath = path.resolve(__dirname, imageFolder, imageName);

  const image = await fs.readFile(imagePath);
  const tensor = tf.node.decodeImage(image);

  const graphModel = await tf.loadGraphModel(modelJSON);
  const model = new automl.ObjectDetectionModel(graphModel, [
    'background',
    'Mucosa',
  ]);

  try {
    const options = { score: 0.2, iou: 0.5, topk: 20 };
    const predictions = await model.detect(tensor, options);
    console.log(predictions);

    predictions.length > 0 &&
      cropImage(
        image,
        imageName,
        Math.round(predictions[0].box.width * 0.5),
        Math.round(predictions[0].box.height * 0.5),
        Math.round(predictions[0].box.left + predictions[0].box.width * 0.25),
        Math.round(predictions[0].box.top + predictions[0].box.height * 0.25)
      );

    return predictions;
  } catch (err) {
    console.log(err);
  }
}

async function readDir(rootDir) {
  rootDir = rootDir || path.resolve(__dirname);

  const files = await fs.readdir(rootDir);

  for (let file of files) {
    await getMucosa('images', file);
  }

  console.log(dominantArray);
}

readDir(path.resolve(__dirname, './images'));

// getColor('cropped_19316_SO_19_09-12-2019_iOS_3015_EDITADO.jpg');
