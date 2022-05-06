const distance = require('ml-distance').distance; // biblioteca para obter Distância Euclidiana
const mode = require('simple-statistics').mode; // biblioteca para obter a Moda
const convert = require('color-convert');

/**
 * Dados com 200 amostras, sendo:
 * - as 100 primeiras de amostras de cor doentes (T)
 * - as 100 últimas amostras de cor saudáveis (NT)
 */
const data = [
  {
    red: 158,
    green: 103,
    blue: 122,
    file: 'cropped_111_SI_21_06-02-2019_iOS_1921.JPG',
    label: 'T',
  },
  {
    red: 193,
    green: 115,
    blue: 121,
    file: 'cropped_111_SI_22_28-11-2018_iOS_0235.JPG',
    label: 'T',
  },
  {
    red: 128,
    green: 85,
    blue: 88,
    file: 'cropped_111_SI_22_28-11-2018_iOS_0235_EDITADO.jpg',
    label: 'T',
  },
  {
    red: 119,
    green: 70,
    blue: 67,
    file: 'cropped_114_SI_21_13-02-2019_iOS_5170.jpg',
    label: 'T',
  },
  {
    red: 185,
    green: 96,
    blue: 94,
    file: 'cropped_116_MN_23_23-01-2019_iOS_1310.JPG',
    label: 'T',
  },
  {
    red: 133,
    green: 80,
    blue: 84,
    file: 'cropped_121_SI_20_13-02-2019_iOS_5192.jpg',
    label: 'T',
  },
  {
    red: 163,
    green: 100,
    blue: 98,
    file: 'cropped_151_SI_21_05-12-2018_iOS_0594.JPG',
    label: 'T',
  },
  {
    red: 130,
    green: 82,
    blue: 93,
    file: 'cropped_15219_SRD_22_21-08-2019_Android_151129.jpg',
    label: 'T',
  },
  {
    red: 185,
    green: 91,
    blue: 76,
    file: 'cropped_15502_SO_20_06-02-2019_iOS_1955.JPG',
    label: 'T',
  },
  {
    red: 202,
    green: 112,
    blue: 91,
    file: 'cropped_15511_SO_17_03-10-2018_iOS_3416.jpg',
    label: 'T',
  },
  {
    red: 206,
    green: 104,
    blue: 87,
    file: 'cropped_15511_SO_23_30-01-2019_iOS_1617.jpg',
    label: 'T',
  },
  {
    red: 201,
    green: 105,
    blue: 94,
    file: 'cropped_15807_SI_16_06-02-2019_iOS_2035.JPG',
    label: 'T',
  },
  {
    red: 105,
    green: 53,
    blue: 69,
    file: 'cropped_15811_SI_22_27-02-2019_iOS_2448.JPG',
    label: 'T',
  },
  {
    red: 191,
    green: 106,
    blue: 113,
    file: 'cropped_162_SI_22_13-02-2019_iOS_5189.jpg',
    label: 'T',
  },
  {
    red: 126,
    green: 67,
    blue: 62,
    file: 'cropped_162_SI_23_28-11-2018_iOS_0229.jpg',
    label: 'T',
  },
  {
    red: 168,
    green: 108,
    blue: 103,
    file: 'cropped_16509_SO_22_30-01-2019_iOS_1710.JPG',
    label: 'T',
  },
  {
    red: 172,
    green: 91,
    blue: 82,
    file: 'cropped_16525_SO_23_12-09-2018_iOS_2366.jpg',
    label: 'T',
  },
  {
    red: 150,
    green: 104,
    blue: 102,
    file: 'cropped_165_SI_23_13-02-2019_Android_083012767.jpg',
    label: 'T',
  },
  {
    red: 151,
    green: 102,
    blue: 113,
    file: 'cropped_17052_MN_16_30-01-2019_iOS_1601.JPG',
    label: 'T',
  },
  {
    red: 102,
    green: 56,
    blue: 64,
    file: 'cropped_1709_SI_21_06-02-2019_iOS_2115.JPG',
    label: 'T',
  },
  {
    red: 200,
    green: 96,
    blue: 82,
    file: 'cropped_17114_SO_23_23-01-2019_iOS_1393.JPG',
    label: 'T',
  },
  {
    red: 126,
    green: 65,
    blue: 64,
    file: 'cropped_17121_SO_22_06-02-2019_iOS_1962.JPG',
    label: 'T',
  },
  {
    red: 127,
    green: 81,
    blue: 83,
    file: 'cropped_1713_SI_21_07-05-2019_iOS_2811.JPG',
    label: 'T',
  },
  {
    red: 107,
    green: 67,
    blue: 89,
    file: 'cropped_1714_SI_23_07-05-2019_iOS_2733.jpg',
    label: 'T',
  },
  {
    red: 157,
    green: 110,
    blue: 99,
    file: 'cropped_1715_SI_21_13-02-2019_Android_090915790.jpg',
    label: 'T',
  },
  {
    red: 193,
    green: 143,
    blue: 131,
    file: 'cropped_1715_SI_21_13-02-2019_Android_090915790_EDITADO.jpg',
    label: 'T',
  },
  {
    red: 170,
    green: 86,
    blue: 77,
    file: 'cropped_1715_SI_23_06-02-2019_iOS_2015.JPG',
    label: 'T',
  },
  {
    red: 169,
    green: 80,
    blue: 83,
    file: 'cropped_17293_SI_23_28-11-2018_iOS_0214.JPG',
    label: 'T',
  },
  {
    red: 203,
    green: 81,
    blue: 49,
    file: 'cropped_17309_SI_22_12-12-2018_iOS_0802.JPG',
    label: 'T',
  },
  {
    red: 168,
    green: 104,
    blue: 107,
    file: 'cropped_177_SI_20_26-09-2018_Android_084533134.jpg',
    label: 'T',
  },
  {
    red: 212,
    green: 88,
    blue: 72,
    file: 'cropped_177_SI_22_06-02-2019_iOS_2073.JPG',
    label: 'T',
  },
  {
    red: 121,
    green: 105,
    blue: 106,
    file: 'cropped_18301_SO_15_21-08-2019_iOS_4750.jpg',
    label: 'T',
  },
  {
    red: 145,
    green: 127,
    blue: 113,
    file: 'cropped_19301_SO_11_09-12-2019_iOS_2903.jpg',
    label: 'T',
  },
  {
    red: 183,
    green: 165,
    blue: 149,
    file: 'cropped_19301_SO_11_09-12-2019_iOS_2903_EDITADO.jpg',
    label: 'T',
  },
  {
    red: 119,
    green: 79,
    blue: 78,
    file: 'cropped_19301_SO_16_20-11-2019_iOS_2735.jpg',
    label: 'T',
  },
  {
    red: 164,
    green: 157,
    blue: 156,
    file: 'cropped_19301_SO_8_14-01-2020_iOS_3159.jpg',
    label: 'T',
  },
  {
    red: 150,
    green: 144,
    blue: 144,
    file: 'cropped_19301_SO_8_14-01-2020_iOS_3159_EDITADO.jpg',
    label: 'T',
  },
  {
    red: 155,
    green: 94,
    blue: 98,
    file: 'cropped_19305_SO_19_09-12-2019_iOS_2927.jpg',
    label: 'T',
  },
  {
    red: 128,
    green: 74,
    blue: 63,
    file: 'cropped_19305_SO_19_14-01-2020_iOS_3140.jpg',
    label: 'T',
  },
  {
    red: 173,
    green: 166,
    blue: 171,
    file: 'cropped_19312_MN_10_09-12-2019_iOS_2970.jpg',
    label: 'T',
  },
  {
    red: 192,
    green: 188,
    blue: 197,
    file: 'cropped_19312_MN_10_09-12-2019_iOS_2970_EDITADO.jpg',
    label: 'T',
  },
  {
    red: 127,
    green: 124,
    blue: 131,
    file: 'cropped_19312_MN_10_09-12-2019_iOS_2970_EDITADO4.jpg',
    label: 'T',
  },
  {
    red: 173,
    green: 152,
    blue: 159,
    file: 'cropped_19315_SO_14_09-12-2019_Android_103022.jpg',
    label: 'T',
  },
  {
    red: 169,
    green: 144,
    blue: 151,
    file: 'cropped_19315_SO_14_09-12-2019_Android_103022_EDITADO.jpg',
    label: 'T',
  },
  {
    red: 187,
    green: 144,
    blue: 146,
    file: 'cropped_19315_SO_15_14-01-2020_iOS_3216.jpg',
    label: 'T',
  },
  {
    red: 202,
    green: 159,
    blue: 161,
    file: 'cropped_19315_SO_15_14-01-2020_iOS_3216_EDITADO.jpg',
    label: 'T',
  },
  {
    red: 173,
    green: 140,
    blue: 118,
    file: 'cropped_19316_SO_10_14-01-2020_iOS_3226.jpg',
    label: 'T',
  },
  {
    red: 173,
    green: 142,
    blue: 119,
    file: 'cropped_19316_SO_10_14-01-2020_iOS_3226__EDITADO.jpg',
    label: 'T',
  },
  {
    red: 107,
    green: 69,
    blue: 61,
    file: 'cropped_19316_SO_19_09-12-2019_iOS_3015.jpg',
    label: 'T',
  },
  {
    red: 154,
    green: 98,
    blue: 86,
    file: 'cropped_19316_SO_19_09-12-2019_iOS_3015_EDITADO.jpg',
    label: 'T',
  },
  {
    red: 202,
    green: 126,
    blue: 116,
    file: 'cropped_19317_SO_21_09-12-2019_iOS_2920.jpg',
    label: 'T',
  },
  {
    red: 174,
    green: 127,
    blue: 121,
    file: 'cropped_19317_SO_21_09-12-2019_iOS_2920_EDITADO.jpg',
    label: 'T',
  },
  {
    red: 124,
    green: 113,
    blue: 109,
    file: 'cropped_19347_MN_9_14-01-2020_iOS_3194.jpg',
    label: 'T',
  },
  {
    red: 115,
    green: 100,
    blue: 96,
    file: 'cropped_19347_MN_9_14-01-2020_iOS_3194_EDITADO.jpg',
    label: 'T',
  },
  {
    red: 174,
    green: 160,
    blue: 155,
    file: 'cropped_19347_MN_9_14-01-2020_iOS_3194_EDITADO1.jpg',
    label: 'T',
  },
  {
    red: 137,
    green: 126,
    blue: 123,
    file: 'cropped_19347_MN_9_14-01-2020_iOS_3194_EDITADO2.jpg',
    label: 'T',
  },
  {
    red: 165,
    green: 129,
    blue: 119,
    file: 'cropped_19353_SO_16_10-10-2019_Android_094037.jpg',
    label: 'T',
  },
  {
    red: 169,
    green: 133,
    blue: 122,
    file: 'cropped_19353_SO_16_10-10-2019_Android_094037_EDITADO.jpg',
    label: 'T',
  },
  {
    red: 150,
    green: 90,
    blue: 89,
    file: 'cropped_19353_SO_23_09-12-2019_iOS_3030.jpg',
    label: 'T',
  },
  {
    red: 189,
    green: 159,
    blue: 152,
    file: 'cropped_19354_SO_11_14-01-2020_iOS_3208.jpg',
    label: 'T',
  },
  {
    red: 179,
    green: 140,
    blue: 132,
    file: 'cropped_19354_SO_14_09-12-2019_iOS_3040.jpg',
    label: 'T',
  },
  {
    red: 174,
    green: 136,
    blue: 130,
    file: 'cropped_19354_SO_14_09-12-2019_iOS_3040_EDITADO.jpg',
    label: 'T',
  },
  {
    red: 80,
    green: 62,
    blue: 62,
    file: 'cropped_19354_SO_17_20-11-2019_iOS_2808.jpg',
    label: 'T',
  },
  {
    red: 82,
    green: 62,
    blue: 63,
    file: 'cropped_19354_SO_17_20-11-2019_iOS_2808_EDITADO.jpg',
    label: 'T',
  },
  {
    red: 177,
    green: 114,
    blue: 90,
    file: 'cropped_19358_MN_20_20-11-2019_iOS_2754.jpg',
    label: 'T',
  },
  {
    red: 134,
    green: 111,
    blue: 108,
    file: 'cropped_19377_SO_11_09-12-2019_iOS_2943.jpg',
    label: 'T',
  },
  {
    red: 136,
    green: 113,
    blue: 110,
    file: 'cropped_19377_SO_11_09-12-2019_iOS_2943_EDITADO.jpg',
    label: 'T',
  },
  {
    red: 188,
    green: 146,
    blue: 128,
    file: 'cropped_19377_SO_16_20-11-2019_iOS_2791.jpg',
    label: 'T',
  },
  {
    red: 141,
    green: 113,
    blue: 100,
    file: 'cropped_19377_SO_16_20-11-2019_iOS_2791_EDITADO.jpg',
    label: 'T',
  },
  {
    red: 121,
    green: 109,
    blue: 105,
    file: 'cropped_19377_SO_19_14-01-2020_iOS_3188.jpg',
    label: 'T',
  },
  {
    red: 132,
    green: 100,
    blue: 93,
    file: 'cropped_19379_SO_11_14-01-2020_iOS_3169.jpg',
    label: 'T',
  },
  {
    red: 211,
    green: 133,
    blue: 119,
    file: 'cropped_19379_SO_15_09-12-2019_iOS_2961.jpg',
    label: 'T',
  },
  {
    red: 109,
    green: 87,
    blue: 86,
    file: 'cropped_19379_SO_18_20-11-2019_iOS_2774.jpg',
    label: 'T',
  },
  {
    red: 184,
    green: 162,
    blue: 171,
    file: 'cropped_19385_SO_13_14-01-2020_Android_102840.jpg',
    label: 'T',
  },
  {
    red: 190,
    green: 170,
    blue: 178,
    file: 'cropped_19385_SO_13_14-01-2020_Android_102840_EDITADO.jpg',
    label: 'T',
  },
  {
    red: 156,
    green: 137,
    blue: 140,
    file: 'cropped_19385_SO_16_09-12-2019_Android_103302.jpg',
    label: 'T',
  },
  {
    red: 123,
    green: 107,
    blue: 109,
    file: 'cropped_19385_SO_16_09-12-2019_Android_103302_EDITADO.jpg',
    label: 'T',
  },
  {
    red: 181,
    green: 135,
    blue: 136,
    file: 'cropped_19391_SO_11_14-01-2020_iOS_3178.jpg',
    label: 'T',
  },
  {
    red: 208,
    green: 175,
    blue: 176,
    file: 'cropped_19396_SO_12_14-01-2020_Android_090734.jpg',
    label: 'T',
  },
  {
    red: 176,
    green: 136,
    blue: 139,
    file: 'cropped_19396_SO_12_14-01-2020_Android_090734_EDITADO.jpg',
    label: 'T',
  },
  {
    red: 146,
    green: 104,
    blue: 99,
    file: 'cropped_19396_SO_17_09-12-2019_iOS_2897.jpg',
    label: 'T',
  },
  {
    red: 113,
    green: 82,
    blue: 77,
    file: 'cropped_19396_SO_17_20-11-2019_iOS_2745.jpg',
    label: 'T',
  },
  {
    red: 155,
    green: 120,
    blue: 121,
    file: 'cropped_19397_SO_12_14-01-2020_Android_103409.jpg',
    label: 'T',
  },
  {
    red: 126,
    green: 94,
    blue: 93,
    file: 'cropped_19397_SO_17_09-12-2019_Android_102801.jpg',
    label: 'T',
  },
  {
    red: 134,
    green: 90,
    blue: 104,
    file: 'cropped_19403_MN_18_10-10-2019_Android_093744.jpg',
    label: 'T',
  },
  {
    red: 136,
    green: 126,
    blue: 113,
    file: 'cropped_19420_SO_10_20-11-2019_Android_100903.jpg',
    label: 'T',
  },
  {
    red: 136,
    green: 126,
    blue: 112,
    file: 'cropped_19420_SO_10_20-11-2019_Android_100903_EDITADO.jpg',
    label: 'T',
  },
  {
    red: 88,
    green: 79,
    blue: 64,
    file: 'cropped_19420_SO_10_20-11-2019_Android_100903_EDITADO2.jpg',
    label: 'T',
  },
  {
    red: 137,
    green: 127,
    blue: 114,
    file: 'cropped_19420_SO_10_20-11-2019_Android_100903_EDITADO3.jpg',
    label: 'T',
  },
  {
    red: 79,
    green: 73,
    blue: 62,
    file: 'cropped_19420_SO_10_20-11-2019_Android_100903_EDITADO4.jpg',
    label: 'T',
  },
  {
    red: 150,
    green: 128,
    blue: 119,
    file: 'cropped_19420_SO_15_09-12-2019_Android_144900.jpg',
    label: 'T',
  },
  {
    red: 159,
    green: 124,
    blue: 122,
    file: 'cropped_19422_SO_22_09-12-2019_iOS_2997.jpg',
    label: 'T',
  },
  {
    red: 157,
    green: 111,
    blue: 95,
    file: 'cropped_19463_SO_14_14-01-2020_iOS_3127.jpg',
    label: 'T',
  },
  {
    red: 160,
    green: 100,
    blue: 102,
    file: 'cropped_309_SI_23_13-02-2019_iOS_4979.jpg',
    label: 'T',
  },
  {
    red: 125,
    green: 87,
    blue: 88,
    file: 'cropped_309_SI_23_13-02-2019_iOS_4979_EDITADO.jpg',
    label: 'T',
  },
  {
    red: 169,
    green: 108,
    blue: 110,
    file: 'cropped_309_SI_23_28-11-2018_Android_084137213.jpg',
    label: 'T',
  },
  {
    red: 170,
    green: 112,
    blue: 109,
    file: 'cropped_361_SI_23_27-08-2018_iOS_1721.jpg',
    label: 'T',
  },
  {
    red: 184,
    green: 86,
    blue: 70,
    file: 'cropped_432_SO_21_06-02-2019_iOS_2067.JPG',
    label: 'T',
  },
  {
    red: 119,
    green: 74,
    blue: 77,
    file: 'cropped_432_SO_21_13-02-2019_iOS_5183.jpg',
    label: 'T',
  },
  {
    red: 118,
    green: 65,
    blue: 81,
    file: 'cropped_721_MN_23_27-02-2019_iOS_2473.jpg',
    label: 'T',
  },

  {
    red: 217,
    green: 85,
    blue: 67,
    file: 'cropped_101_SI_25_30-01-2019_iOS_1575.JPG',
    label: 'NT',
  },
  {
    red: 223,
    green: 98,
    blue: 79,
    file: 'cropped_101_SI_24_23-01-2019_iOS_1187.JPG',
    label: 'NT',
  },
  {
    red: 178,
    green: 81,
    blue: 83,
    file: 'cropped_101_SI_26_13-02-2019_iOS_5102.jpg',
    label: 'NT',
  },
  {
    red: 185,
    green: 124,
    blue: 118,
    file: 'cropped_101_SI_28_24-10-2018_Android_083228655.jpg',
    label: 'NT',
  },
  {
    red: 207,
    green: 113,
    blue: 117,
    file: 'cropped_101_SI_29_03-10-2018_Android_082042169.jpg',
    label: 'NT',
  },
  {
    red: 175,
    green: 98,
    blue: 94,
    file: 'cropped_109_SI_29_17-10-2018_iOS_3939.jpg',
    label: 'NT',
  },
  {
    red: 150,
    green: 77,
    blue: 72,
    file: 'cropped_109_SI_30_24-10-2018_Android_090159682.jpg',
    label: 'NT',
  },
  {
    red: 194,
    green: 94,
    blue: 90,
    file: 'cropped_111_SI_24_23-01-2019_iOS_1378.JPG',
    label: 'NT',
  },
  {
    red: 70,
    green: 35,
    blue: 36,
    file: 'cropped_111_SI_24_24-10-2018_Android_082349624.jpg',
    label: 'NT',
  },
  {
    red: 195,
    green: 82,
    blue: 65,
    file: 'cropped_114_SI_25_05-12-2018_iOS_0611.JPG',
    label: 'NT',
  },
  {
    red: 180,
    green: 130,
    blue: 125,
    file: 'cropped_114_SI_25_07-11-2018_Android_082405.jpg',
    label: 'NT',
  },
  {
    red: 173,
    green: 83,
    blue: 94,
    file: 'cropped_115_SI_25_24-10-2018_iOS_4215.jpg',
    label: 'NT',
  },
  {
    red: 166,
    green: 83,
    blue: 80,
    file: 'cropped_115_SI_26_12-12-2018_Android_090452566.jpg',
    label: 'NT',
  },
  {
    red: 230,
    green: 173,
    blue: 191,
    file: 'cropped_115_SI_27_17-10-2018_Android_094645112.jpg',
    label: 'NT',
  },
  {
    red: 184,
    green: 75,
    blue: 55,
    file: 'cropped_115_SI_27_30-01-2019_iOS_1526.JPG',
    label: 'NT',
  },
  {
    red: 147,
    green: 81,
    blue: 96,
    file: 'cropped_116_MN_24_30-01-2019_iOS_1569.JPG',
    label: 'NT',
  },
  {
    red: 176,
    green: 120,
    blue: 137,
    file: 'cropped_116_MN_28_24-10-2018_iOS_4334.jpg',
    label: 'NT',
  },
  {
    red: 210,
    green: 111,
    blue: 104,
    file: 'cropped_117_SI_24_06-02-2019_iOS_1841.JPG',
    label: 'NT',
  },
  {
    red: 192,
    green: 100,
    blue: 99,
    file: 'cropped_117_SI_26_05-12-2018_iOS_0492.JPG',
    label: 'NT',
  },
  {
    red: 136,
    green: 72,
    blue: 69,
    file: 'cropped_117_SI_26_12-12-2018_Android_090338048.jpg',
    label: 'NT',
  },
  {
    red: 161,
    green: 73,
    blue: 61,
    file: 'cropped_117_SI_30_14-11-2018_iOS_5294.jpg',
    label: 'NT',
  },
  {
    red: 190,
    green: 118,
    blue: 106,
    file: 'cropped_117_SI_32_17-10-2018_Android_083805113.jpg',
    label: 'NT',
  },
  {
    red: 125,
    green: 73,
    blue: 68,
    file: 'cropped_118_MN_25_05-12-2018_iOS_0519.JPG',
    label: 'NT',
  },
  {
    red: 235,
    green: 129,
    blue: 115,
    file: 'cropped_121_SI_27_27-08-2018_iOS_1752.jpg',
    label: 'NT',
  },
  {
    red: 159,
    green: 119,
    blue: 129,
    file: 'cropped_121_SI_30_05-09-2018_Android_093928759.jpg',
    label: 'NT',
  },
  {
    red: 111,
    green: 52,
    blue: 54,
    file: 'cropped_123_SI_26_06-02-2019_iOS_1875.JPG',
    label: 'NT',
  },
  {
    red: 181,
    green: 108,
    blue: 142,
    file: 'cropped_123_SI_27_13-02-2019_iOS_5197.jpg',
    label: 'NT',
  },
  {
    red: 166,
    green: 90,
    blue: 104,
    file: 'cropped_123_SI_29_14-11-2018_iOS_5320.jpg',
    label: 'NT',
  },
  {
    red: 188,
    green: 109,
    blue: 115,
    file: 'cropped_123_SI_30_07-11-2018_Android_085702.jpg',
    label: 'NT',
  },
  {
    red: 210,
    green: 83,
    blue: 69,
    file: 'cropped_129_SI_27_05-12-2018_iOS_0687.JPG',
    label: 'NT',
  },
  {
    red: 211,
    green: 121,
    blue: 115,
    file: 'cropped_129_SI_29_06-02-2019_iOS_1805.JPG',
    label: 'NT',
  },
  {
    red: 219,
    green: 101,
    blue: 92,
    file: 'cropped_129_SI_30_24-10-2018_iOS_4220.jpg',
    label: 'NT',
  },
  {
    red: 223,
    green: 137,
    blue: 141,
    file: 'cropped_149_SI_34_30-01-2019_iOS_1659.JPG',
    label: 'NT',
  },
  {
    red: 154,
    green: 67,
    blue: 58,
    file: 'cropped_150_SI_24_06-02-2019_iOS_1844.JPG',
    label: 'NT',
  },
  {
    red: 189,
    green: 70,
    blue: 53,
    file: 'cropped_151_SI_29_30-01-2019_iOS_1550.jpg',
    label: 'NT',
  },
  {
    red: 181,
    green: 83,
    blue: 93,
    file: 'cropped_155_SI_38_03-10-2018_Android_081749169.jpg',
    label: 'NT',
  },
  {
    red: 172,
    green: 76,
    blue: 80,
    file: 'cropped_15804_SI_34_10-10-2018_Android_084520313.jpg',
    label: 'NT',
  },
  {
    red: 186,
    green: 108,
    blue: 116,
    file: 'cropped_15852_SI_27_12-09-2018_Android_082838985.jpg',
    label: 'NT',
  },
  {
    red: 103,
    green: 46,
    blue: 59,
    file: 'cropped_166_MN_30_26-09-2018_Android_081407839.jpg',
    label: 'NT',
  },
  {
    red: 180,
    green: 104,
    blue: 118,
    file: 'cropped_166_MN_28_03-10-2018_iOS_3393.jpg',
    label: 'NT',
  },
  {
    red: 155,
    green: 90,
    blue: 108,
    file: 'cropped_166_MN_31_19-09-2018_Android_1189924240.jpg',
    label: 'NT',
  },
  {
    red: 152,
    green: 71,
    blue: 65,
    file: 'cropped_16970_SI_32_27-08-2018_iOS_1728.jpg',
    label: 'NT',
  },
  {
    red: 115,
    green: 63,
    blue: 68,
    file: 'cropped_16970_SI_35_20-02-2019_Android_083902871.jpg',
    label: 'NT',
  },
  {
    red: 178,
    green: 104,
    blue: 102,
    file: 'cropped_17111_MN_26_30-01-2019_iOS_1696.jpg',
    label: 'NT',
  },
  {
    red: 173,
    green: 106,
    blue: 114,
    file: 'cropped_17116_SO_27_06-02-2019_iOS_2122.JPG',
    label: 'NT',
  },
  {
    red: 129,
    green: 64,
    blue: 63,
    file: 'cropped_17261_SI_24_06-02-2019_Android_090936583.jpg',
    label: 'NT',
  },
  {
    red: 173,
    green: 68,
    blue: 58,
    file: 'cropped_17313_SI_29_12-12-2018_iOS_0847.JPG',
    label: 'NT',
  },
  {
    red: 180,
    green: 95,
    blue: 97,
    file: 'cropped_17313_SI_30_28-11-2018_iOS_0256.JPG',
    label: 'NT',
  },
  {
    red: 166,
    green: 101,
    blue: 103,
    file: 'cropped_177_MN_27_07-05-2019_iOS_2841.JPG',
    label: 'NT',
  },
  {
    red: 170,
    green: 88,
    blue: 78,
    file: 'cropped_177_SI_24_28-11-2018_Android_090406268.jpg',
    label: 'NT',
  },
  {
    red: 153,
    green: 90,
    blue: 89,
    file: 'cropped_178_SI_30_06-02-2019_iOS_2103.JPG',
    label: 'NT',
  },
  {
    red: 187,
    green: 115,
    blue: 115,
    file: 'cropped_179_SI_27_30-01-2019_iOS_1642.JPG',
    label: 'NT',
  },
  {
    red: 121,
    green: 84,
    blue: 97,
    file: 'cropped_182_MN_26_26-09-2018_Android_082513901.jpg',
    label: 'NT',
  },
  {
    red: 207,
    green: 120,
    blue: 115,
    file: 'cropped_180_MN_28_30-01-2019_Android_083600608.jpg',
    label: 'NT',
  },
  {
    red: 161,
    green: 98,
    blue: 111,
    file: 'cropped_183_MN_26_06-02-2019_iOS_1855.JPG',
    label: 'NT',
  },
  {
    red: 166,
    green: 130,
    blue: 163,
    file: 'cropped_183_MN_28_19-09-2018_Android_939844585.jpg',
    label: 'NT',
  },
  {
    red: 104,
    green: 60,
    blue: 73,
    file: 'cropped_183_MN_30_20-02-2019_Android_090212355.jpg',
    label: 'NT',
  },
  {
    red: 128,
    green: 52,
    blue: 51,
    file: 'cropped_185_SI_28_14-11-2018_Android_085800933.jpg',
    label: 'NT',
  },
  {
    red: 140,
    green: 76,
    blue: 74,
    file: 'cropped_198_SI_30_10-10-2018_Android_092100358.jpg',
    label: 'NT',
  },
  {
    red: 148,
    green: 98,
    blue: 108,
    file: 'cropped_278_MN_31_19-09-2018_Android_191119982.jpg',
    label: 'NT',
  },
  {
    red: 93,
    green: 61,
    blue: 79,
    file: 'cropped_278_MN_35_10-10-2018_Android_090513361.jpg',
    label: 'NT',
  },
  {
    red: 139,
    green: 98,
    blue: 116,
    file: 'cropped_279_MN_29_26-09-2018_Android_081223122.jpg',
    label: 'NT',
  },
  {
    red: 192,
    green: 90,
    blue: 69,
    file: 'cropped_279_MN_33_23-01-2019_iOS_1485.jpg',
    label: 'NT',
  },
  {
    red: 184,
    green: 126,
    blue: 143,
    file: 'cropped_307_MN_39_10-10-2018_Android_084617613.jpg',
    label: 'NT',
  },
  {
    red: 146,
    green: 98,
    blue: 92,
    file: 'cropped_307_MN_31_21-11-2018_Android_092818086.jpg',
    label: 'NT',
  },
  {
    red: 145,
    green: 67,
    blue: 59,
    file: 'cropped_308_SO_31_26-09-2018_Android_090228120.jpg',
    label: 'NT',
  },
  {
    red: 226,
    green: 175,
    blue: 164,
    file: 'cropped_309_SO_28_19-09-2018_Android_1540715561.jpg',
    label: 'NT',
  },
  {
    red: 138,
    green: 72,
    blue: 66,
    file: 'cropped_309_SO_28_05-12-2018_Android_083517459.jpg',
    label: 'NT',
  },
  {
    red: 135,
    green: 96,
    blue: 113,
    file: 'cropped_312_MN_30_12-12-2018_Android_085220350.jpg',
    label: 'NT',
  },
  {
    red: 161,
    green: 113,
    blue: 142,
    file: 'cropped_312_MN_30_19-09-2018_Android_626673893.jpg',
    label: 'NT',
  },
  {
    red: 96,
    green: 54,
    blue: 59,
    file: 'cropped_320_MN_26_05-09-2018_Android_083342394.jpg',
    label: 'NT',
  },
  {
    red: 125,
    green: 65,
    blue: 80,
    file: 'cropped_320_MN_28_06-02-2019_iOS_1868.JPG',
    label: 'NT',
  },
  {
    red: 153,
    green: 97,
    blue: 84,
    file: 'cropped_320_MN_31_21-11-2018_Android_090621468.jpg',
    label: 'NT',
  },
  {
    red: 167,
    green: 104,
    blue: 119,
    file: 'cropped_323_MN_28_17-10-2018_iOS_3988.jpg',
    label: 'NT',
  },
  {
    red: 103,
    green: 71,
    blue: 78,
    file: 'cropped_344_MN_34_05-09-2018_Android_085230430.jpg',
    label: 'NT',
  },
  {
    red: 158,
    green: 85,
    blue: 88,
    file: 'cropped_323_MN_30_14-11-2018_iOS_5309.jpg',
    label: 'NT',
  },
  {
    red: 178,
    green: 99,
    blue: 84,
    file: 'cropped_344_SI_26_21-11-2018_Android_092528918.jpg',
    label: 'NT',
  },
  {
    red: 160,
    green: 89,
    blue: 108,
    file: 'cropped_344_MN_35_17-10-2018_iOS_3834.jpg',
    label: 'NT',
  },
  {
    red: 165,
    green: 78,
    blue: 85,
    file: 'cropped_344_SI_34_14-11-2018_iOS_5299.jpg',
    label: 'NT',
  },
  {
    red: 156,
    green: 138,
    blue: 156,
    file: 'cropped_347_MN_24_05-09-2018_Android_083525373.jpg',
    label: 'NT',
  },
  {
    red: 188,
    green: 118,
    blue: 113,
    file: 'cropped_354_MN_26_12-12-2018_Android_083326979.jpg',
    label: 'NT',
  },
  {
    red: 86,
    green: 55,
    blue: 63,
    file: 'cropped_354_MN_26_20-02-2019_Android_083617257.jpg',
    label: 'NT',
  },
  {
    red: 124,
    green: 60,
    blue: 55,
    file: 'cropped_354_MN_28_07-05-2019_iOS_2883.jpg',
    label: 'NT',
  },
  {
    red: 147,
    green: 72,
    blue: 77,
    file: 'cropped_355_MN_31_23-01-2019_iOS_1322.JPG',
    label: 'NT',
  },
  {
    red: 111,
    green: 57,
    blue: 69,
    file: 'cropped_355_MN_31_26-09-2018_iOS_3002.jpg',
    label: 'NT',
  },
  {
    red: 112,
    green: 83,
    blue: 96,
    file: 'cropped_355_MN_32_20-02-2019_Android_082616623.jpg',
    label: 'NT',
  },
  {
    red: 161,
    green: 117,
    blue: 141,
    file: 'cropped_359_MN_36_07-11-2018_Android_085218.jpg',
    label: 'NT',
  },
  {
    red: 146,
    green: 80,
    blue: 87,
    file: 'cropped_360_SI_28_24-10-2018_iOS_4206.jpg',
    label: 'NT',
  },
  {
    red: 179,
    green: 92,
    blue: 82,
    file: 'cropped_361_SI_29_12-09-2018_Android_095019491.jpg',
    label: 'NT',
  },
  {
    red: 119,
    green: 71,
    blue: 81,
    file: 'cropped_376_MN_25_07-05-2019_iOS_2775.jpg',
    label: 'NT',
  },
  {
    red: 178,
    green: 136,
    blue: 153,
    file: 'cropped_376_MN_28_17-10-2018_Android_095812870.jpg',
    label: 'NT',
  },
  {
    red: 156,
    green: 108,
    blue: 127,
    file: 'cropped_376_MN_30_12-09-2018_Android_094642188.jpg',
    label: 'NT',
  },
  {
    red: 213,
    green: 109,
    blue: 119,
    file: 'cropped_376_MN_31_03-10-2018_iOS_3518.jpg',
    label: 'NT',
  },
  {
    red: 108,
    green: 68,
    blue: 83,
    file: 'cropped_387_SI_24_13-02-2019_iOS_5078.jpg',
    label: 'NT',
  },
  {
    red: 105,
    green: 48,
    blue: 54,
    file: 'cropped_387_SI_26_24-10-2018_Android_085239398.jpg',
    label: 'NT',
  },
  {
    red: 129,
    green: 51,
    blue: 39,
    file: 'cropped_395_MN_24_12-12-2018_iOS_0853.JPG',
    label: 'NT',
  },
  {
    red: 177,
    green: 107,
    blue: 92,
    file: 'cropped_503_MN_29_12-12-2018_Android_091446305.jpg',
    label: 'NT',
  },
  {
    red: 174,
    green: 112,
    blue: 130,
    file: 'cropped_395_MN_30_03-10-2018_iOS_3512.jpg',
    label: 'NT',
  },
  {
    red: 190,
    green: 118,
    blue: 106,
    file: 'cropped_117_SI_32_17-10-2018_Android_083805113.jpg',
    label: 'NT',
  },
  {
    red: 235,
    green: 129,
    blue: 115,
    file: 'cropped_121_SI_27_27-08-2018_iOS_1752.jpg',
    label: 'NT',
  },
];

function knn(corRGB, trainingData, n) {
  let values = { red: corRGB.red, green: corRGB.green, blue: corRGB.blue };
  const k = n;
  const data = [...trainingData];

  const classe = data.map((info) => info.label);

  let obj = data.map((info) => {
    let aux = [];
    aux.push(info.red);
    aux.push(info.green);
    aux.push(info.blue);
    return aux;
  });

  // valores máximos de cada coluna
  const max0 = Math.max.apply(
    null,
    obj.map((a, i) => a[0])
  );
  const max1 = Math.max.apply(
    null,
    obj.map((a, i) => a[1])
  );
  const max2 = Math.max.apply(
    null,
    obj.map((a, i) => a[2])
  );

  // normalizando entrada
  // values = [values[0] / max0, values[1] / max1, values[2] / max2];
  values = [values.red / max0, values.green / max1, values.blue / max2];

  // normalizando vetor
  obj = obj.map((a, i) => {
    a[0] = a[0] / max0;
    a[1] = a[1] / max1;
    a[2] = a[2] / max2;

    return a;
  });

  // obtendo distância euclidiana
  const dist = obj.map((a, i) => distance.euclidean(a, values));

  // Ordenando as distâncias do menor para maior
  var distwithlabel = dist.map((a, i) => [a, classe[i]]);
  distwithlabel = distwithlabel.sort((a, b) => {
    if (a[0] < b[0]) return -1;
    if (a[0] > b[0]) return 1;
    return 0;
  });

  let knearest = distwithlabel.slice(0, k);

  let labels = knearest.map((el) => el[1]);

  // retornando apenas a classificação dos K primeiro elementos que mais se repete
  // return mode(distwithlabel.slice(0, k));
  return mode(labels);
}

function lmknnRGB(corRGB, trainingData, n) {
  let values = { red: corRGB.red, green: corRGB.green, blue: corRGB.blue };
  const k = n;
  const data = [...trainingData];

  const classe = data.map((info) => info.label);

  let obj = data.map((info) => {
    let aux = [];
    aux.push(info.red);
    aux.push(info.green);
    aux.push(info.blue);
    return aux;
  });

  // // valores máximos de cada coluna
  // const max0 = Math.max.apply(
  //   null,
  //   obj.map((a, i) => a[0])
  // );
  // const max1 = Math.max.apply(
  //   null,
  //   obj.map((a, i) => a[1])
  // );
  // const max2 = Math.max.apply(
  //   null,
  //   obj.map((a, i) => a[2])
  // );

  // // normalizando entrada
  // // values = [values[0] / max0, values[1] / max1, values[2] / max2];
  // values = [values.red / max0, values.green / max1, values.blue / max2];
  values = [values.red, values.green, values.blue];

  // // normalizando vetor
  // obj = obj.map((a, i) => {
  //   a[0] = a[0] / max0;
  //   a[1] = a[1] / max1;
  //   a[2] = a[2] / max2;

  //   return a;
  // });

  // obtendo distância euclidiana
  const dist = obj.map((a, i) => distance.euclidean(a, values));

  // Ordenando as distâncias do menor para maior
  let distwithlabel = dist.map((dist, i) => [dist, classe[i], obj[i]]);
  distwithlabel = distwithlabel.sort((a, b) => {
    if (a[0] < b[0]) return -1;
    if (a[0] > b[0]) return 1;
    return 0;
  });

  //pega os os k vetores mais próximos com classe T
  let kObjectsWithTClass = [];
  let i = 0;
  for (let dist of distwithlabel) {
    if (dist[1] === 'T') {
      kObjectsWithTClass.push(dist);
      i++;
    }
    if (i === k) break;
  }

  // pega a soma total da primeira posição de todos os vetores da classe T
  let sum00T = kObjectsWithTClass.reduce(
    (previousValue, currentValue) => previousValue + currentValue[2][0],
    0
  );

  // pega a soma total da segunda posição de todos os vetores da classe T
  let sum01T = kObjectsWithTClass.reduce(
    (previousValue, currentValue) => previousValue + currentValue[2][1],
    0
  );

  // pega a soma total da terceira posição de todos os vetores da classe T
  let sum02T = kObjectsWithTClass.reduce(
    (previousValue, currentValue) => previousValue + currentValue[2][2],
    0
  );
  let mean00T = sum00T / k;
  let mean01T = sum01T / k;
  let mean02T = sum02T / k;
  let meanObjectsWithTClass = [mean00T, mean01T, mean02T]; //centroid
  let euclidiandDistanteFromTCentroid = distance.euclidean(
    meanObjectsWithTClass,
    values
  );

  //pega os os k vetores mais próximos com classe NT
  let kObjectsWithNTClass = [];
  i = 0;
  for (let dist of distwithlabel) {
    if (dist[1] === 'NT') {
      kObjectsWithNTClass.push(dist);
      i++;
    }
    if (i === k) break;
  }
  let sum00NT = kObjectsWithNTClass.reduce(
    (previousValue, currentValue) => previousValue + currentValue[2][0],
    0
  );
  let sum01NT = kObjectsWithNTClass.reduce(
    (previousValue, currentValue) => previousValue + currentValue[2][1],
    0
  );
  let sum02NT = kObjectsWithNTClass.reduce(
    (previousValue, currentValue) => previousValue + currentValue[2][2],
    0
  );
  mean00NT = sum00NT / k;
  mean01NT = sum01NT / k;
  mean02NT = sum02NT / k;
  let meanObjectsWithNTClass = [mean00NT, mean01NT, mean02NT]; //centroid
  let euclidiandDistanteFromNTCentroid = distance.euclidean(
    meanObjectsWithNTClass,
    values
  );

  // console.log('Distance centroid T:', euclidiandDistanteFromTCentroid);
  // console.log('Distance centroid NT:', euclidiandDistanteFromNTCentroid);
  // console.log('');
  return euclidiandDistanteFromTCentroid < euclidiandDistanteFromNTCentroid
    ? 'T'
    : 'NT';
}

function lmknnCMYK(corCMYK, trainingData, n) {
  let values = { c: corCMYK.c, m: corCMYK.m, y: corCMYK.y, k: corCMYK.k };
  const k = n;
  const data = [...trainingData];

  const classe = data.map((info) => info.label);

  let obj = data.map((info) => {
    let aux = [];
    aux.push(info.c);
    aux.push(info.m);
    aux.push(info.y);
    aux.push(info.k);
    return aux;
  });

  // // valores máximos de cada coluna
  // const max0 = Math.max.apply(
  //   null,
  //   obj.map((a, i) => a[0])
  // );
  // const max1 = Math.max.apply(
  //   null,
  //   obj.map((a, i) => a[1])
  // );
  // const max2 = Math.max.apply(
  //   null,
  //   obj.map((a, i) => a[2])
  // );

  // // normalizando entrada
  // // values = [values[0] / max0, values[1] / max1, values[2] / max2];
  // values = [values.red / max0, values.green / max1, values.blue / max2];
  values = [values.c, values.m, values.y, values.k];

  // // normalizando vetor
  // obj = obj.map((a, i) => {
  //   a[0] = a[0] / max0;
  //   a[1] = a[1] / max1;
  //   a[2] = a[2] / max2;

  //   return a;
  // });

  // obtendo distância euclidiana
  const dist = obj.map((a, i) => distance.euclidean(a, values));

  // Ordenando as distâncias do menor para maior
  let distwithlabel = dist.map((dist, i) => [dist, classe[i], obj[i]]);
  distwithlabel = distwithlabel.sort((a, b) => {
    if (a[0] < b[0]) return -1;
    if (a[0] > b[0]) return 1;
    return 0;
  });

  //pega os os k vetores mais próximos com classe T
  let kObjectsWithTClass = [];
  let i = 0;
  for (let dist of distwithlabel) {
    if (dist[1] === 'T') {
      kObjectsWithTClass.push(dist);
      i++;
    }
    if (i === k) break;
  }

  // pega a soma total da primeira posição de todos os vetores da classe T
  let sum00T = kObjectsWithTClass.reduce(
    (previousValue, currentValue) => previousValue + currentValue[2][0],
    0
  );

  // pega a soma total da segunda posição de todos os vetores da classe T
  let sum01T = kObjectsWithTClass.reduce(
    (previousValue, currentValue) => previousValue + currentValue[2][1],
    0
  );

  // pega a soma total da terceira posição de todos os vetores da classe T
  let sum02T = kObjectsWithTClass.reduce(
    (previousValue, currentValue) => previousValue + currentValue[2][2],
    0
  );

  // pega a soma total da quarta posição de todos os vetores da classe T
  let sum03T = kObjectsWithTClass.reduce(
    (previousValue, currentValue) => previousValue + currentValue[2][3],
    0
  );

  let mean00T = sum00T / k;
  let mean01T = sum01T / k;
  let mean02T = sum02T / k;
  let mean03T = sum03T / k;
  let meanObjectsWithTClass = [mean00T, mean01T, mean02T, mean03T]; //centroid
  let euclidiandDistanteFromTCentroid = distance.euclidean(
    meanObjectsWithTClass,
    values
  );

  //pega os os k vetores mais próximos com classe NT
  let kObjectsWithNTClass = [];
  i = 0;
  for (let dist of distwithlabel) {
    if (dist[1] === 'NT') {
      kObjectsWithNTClass.push(dist);
      i++;
    }
    if (i === k) break;
  }
  let sum00NT = kObjectsWithNTClass.reduce(
    (previousValue, currentValue) => previousValue + currentValue[2][0],
    0
  );
  let sum01NT = kObjectsWithNTClass.reduce(
    (previousValue, currentValue) => previousValue + currentValue[2][1],
    0
  );
  let sum02NT = kObjectsWithNTClass.reduce(
    (previousValue, currentValue) => previousValue + currentValue[2][2],
    0
  );
  let sum03NT = kObjectsWithNTClass.reduce(
    (previousValue, currentValue) => previousValue + currentValue[2][3],
    0
  );
  mean00NT = sum00NT / k;
  mean01NT = sum01NT / k;
  mean02NT = sum02NT / k;
  mean03NT = sum03NT / k;
  let meanObjectsWithNTClass = [mean00NT, mean01NT, mean02NT, mean03NT]; //centroid
  let euclidiandDistanteFromNTCentroid = distance.euclidean(
    meanObjectsWithNTClass,
    values
  );

  // console.log('Distance centroid T:', euclidiandDistanteFromTCentroid);
  // console.log('Distance centroid NT:', euclidiandDistanteFromNTCentroid);
  // console.log('');
  return euclidiandDistanteFromTCentroid < euclidiandDistanteFromNTCentroid
    ? 'T'
    : 'NT';
}

function lmknnPlusDwknn(corRGB, trainingData) {
  let values = { red: corRGB.red, green: corRGB.green, blue: corRGB.blue };
  const k = 15;
  const data = [...trainingData];

  const classe = data.map((info) => info.label);

  let obj = data.map((info) => {
    let aux = [];
    aux.push(info.red);
    aux.push(info.green);
    aux.push(info.blue);
    return aux;
  });

  // // valores máximos de cada coluna
  // const max0 = Math.max.apply(
  //   null,
  //   obj.map((a, i) => a[0])
  // );
  // const max1 = Math.max.apply(
  //   null,
  //   obj.map((a, i) => a[1])
  // );
  // const max2 = Math.max.apply(
  //   null,
  //   obj.map((a, i) => a[2])
  // );

  // // normalizando entrada
  // // values = [values[0] / max0, values[1] / max1, values[2] / max2];
  // values = [values.red / max0, values.green / max1, values.blue / max2];
  values = [values.red, values.green, values.blue];

  // // normalizando vetor
  // obj = obj.map((a, i) => {
  //   a[0] = a[0] / max0;
  //   a[1] = a[1] / max1;
  //   a[2] = a[2] / max2;

  //   return a;
  // });

  // obtendo distância euclidiana
  const dist = obj.map((a, i) => distance.euclidean(a, values));

  // Ordenando as distâncias do menor para maior
  let distwithlabel = dist.map((dist, i) => [dist, classe[i], obj[i]]);
  distwithlabel = distwithlabel.sort((a, b) => {
    if (a[0] < b[0]) return -1;
    if (a[0] > b[0]) return 1;
    return 0;
  });

  //pega os os k vetores mais próximos com classe T
  let kObjectsWithTClass = [];
  let i = 0;
  for (let dist of distwithlabel) {
    if (dist[1] === 'T') {
      kObjectsWithTClass.push(dist);
      i++;
    }
    if (i === k) break;
  }

  //pega os os k vetores mais próximos com classe NT
  let kObjectsWithNTClass = [];
  i = 0;
  for (let dist of distwithlabel) {
    if (dist[1] === 'NT') {
      kObjectsWithNTClass.push(dist);
      i++;
    }
    if (i === k) break;
  }

  // calcula os pesos
  let kWeightsTClass = kObjectsWithTClass.map((vector) =>
    vector[0] > 0 ? 1 / vector[0] : 1
  );
  let kWeightsNTClass = kObjectsWithNTClass.map((vector) =>
    vector[0] > 0 ? 1 / vector[0] : 1
  );

  // calcula as médias dos pesos de cada classe
  let TWeightMean = kWeightsTClass.reduce(
    (previousValue, currentValue) => previousValue + currentValue,
    0
  );
  TWeightMean = TWeightMean / k;

  let NTWeightMean = kWeightsNTClass.reduce(
    (previousValue, currentValue) => previousValue + currentValue,
    0
  );
  NTWeightMean = NTWeightMean / k;

  // console.log('Peso T:', TWeightMean);
  // console.log('Peso NT:', NTWeightMean);
  // console.log('');
  return TWeightMean > NTWeightMean ? 'T' : 'NT';
}

function getAccuracy(t = 10, fullData, k = 1) {
  let trainingData, helthTestData, sickTestData;
  const dataLength = fullData.length;
  const halfDataLength = dataLength / 2;

  let T = 0;
  let NT = 0;
  for (let d of data) {
    if (d.label === 'T') T++;
    if (d.label === 'NT') NT++;
  }
  // console.log('Quantidade T:', T);
  // console.log('Quantidade NT:', NT);

  let accuracyArr = [];
  let meanAccuracy = 0;

  let hitsArr = [];

  for (let i = 0; i < t; i++) {
    trainingData = [...fullData];

    let sickClassification = [];
    let healthyClassification = [];
    let hits = 0;
    let accuracy;

    /**
     * Seleciona as k amostras de doentes (que vão de 0 até a metade)
     */
    sickTestData = trainingData.splice(i * t, t);
    // console.log('Doentes:', sickTestData);

    /**
     * Seleciona as k amostras doentes (que vão da metada em diante),
     * obs.: desconsidera as k amostras removidas no splice anterior
     */
    helthTestData = trainingData.splice(i * t + halfDataLength - t, t);
    // console.log('Saudáveis:', helthTestData);

    /**
     * Classifica as amostras de teste e salva em um vetor
     */
    sickClassification = sickTestData.map(
      (element) => lmknnRGB(element, trainingData, k)
      // (element) => lmknnCMYK(element, trainingData, k)
      // knn(element, trainingData, k)
    );
    // console.log('Classificação doentes:', sickClassification);

    healthyClassification = helthTestData.map(
      (element) => lmknnRGB(element, trainingData, k)
      // (element) => lmknnCMYK(element, trainingData, k)
      // knn(element, trainingData, k)
    );
    // console.log('Classificação dos saudáveis:', healthyClassification);

    /**
     * conta a quantidade de acertos
     */
    sickClassification.forEach((classification, i) => {
      if (classification === sickTestData[i].label) hits++;
    });

    healthyClassification.forEach((classification, i) => {
      if (classification === helthTestData[i].label) hits++;
    });

    /**
     * Calcula a acurácia
     */
    accuracy = hits / (t * 2);
    accuracyArr.push(accuracy);
    // console.log('quantidade de acertos:', hits);
    // console.log(`Acurárica ${i + 1}:`, accuracy);

    hitsArr.push(hits);
  }
  // console.log('Array de acurácias: ', accuracyArr);

  let sumAccuracies = 0;
  accuracyArr.forEach((el) => (sumAccuracies += Number(el)));
  // console.log(sumAccuracies);

  meanAccuracy = (sumAccuracies / t).toFixed(2);
  console.log(`Acurácia média k=${k}:`, meanAccuracy);
  // console.log('');

  /**
   * Calculo do desvio padrão
   */
  let xMean =
    hitsArr.reduce(
      (previousValue, currentValue) => previousValue + currentValue,
      0
    ) / 10;
  let sumXLessXMean = hitsArr.reduce(
    (previousValue, currentValue) =>
      previousValue + (currentValue - xMean) ** 2,
    0
  );

  let standardDeviation = Math.sqrt(sumXLessXMean / (10 - 1));

  // console.log('Array da quantidade de acertos:', hitsArr);
  // console.log('Média dos acertos:', xMean);
  // console.log(
  //   'Somatório dos acertos menos medias dos acertos ao quadrado:',
  //   sumXLessXMean
  // );
  // console.log('Desvio padrão: ', standardDeviation);
}

// getAccuracy(10, data);

function getPoints(colorArr) {
  let redT = [];
  colorArr.forEach((e) => {
    if (e.label === 'T') redT.push(e.red);
  });

  let greenT = [];
  colorArr.forEach((e) => {
    if (e.label === 'T') greenT.push(e.green);
  });

  let blueT = [];
  colorArr.forEach((e) => {
    if (e.label === 'T') blueT.push(e.blue);
  });
  console.log('redT:', redT);
  console.log('greenT', greenT);
  console.log('blueT', blueT);

  let redNT = [];
  colorArr.forEach((e) => {
    if (e.label === 'NT') redNT.push(e.red);
  });

  let greenNT = [];
  colorArr.forEach((e) => {
    if (e.label === 'NT') greenNT.push(e.green);
  });

  let blueNT = [];
  colorArr.forEach((e) => {
    if (e.label === 'NT') blueNT.push(e.blue);
  });
  console.log('redNT:', redNT);
  console.log('greenNT', greenNT);
  console.log('blueNT', blueNT);
}

let dataHSL = data.map((data) => {
  let hsl = convert.rgb.hsl([data.red, data.green, data.blue]);
  return {
    red: hsl[0],
    green: hsl[1],
    blue: hsl[2],
    file: data.file,
    label: data.label,
  };
});

// getPoints(dataHSL);
for (let i = 1; i <= 100; i++) {
  getAccuracy(10, dataHSL, i);
}

// let dataCMYK = data.map((data) => {
//   let cmyk = convert.rgb.cmyk(data.red, data.green, data.blue);
//   return {
//     c: cmyk[0],
//     m: cmyk[1],
//     y: cmyk[2],
//     k: cmyk[3],
//     file: data.file,
//     label: data.label,
//   };
// });
// // console.log(dataCMYK);

// for (let i = 1; i <= 100; i++) {
//   getAccuracy(10, dataCMYK, i);
// }
