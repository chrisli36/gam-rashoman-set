
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import re
import numpy as np
import pandas as pd

header = ['sex=female<=0', '0<sex=female<=1', 'age<=18', '18<age<=19', '19<age<=20', '20<age<=21', '21<age<=22', '22<age<=23', '23<age<=24', '24<age<=25', '25<age<=26', '26<age<=27', '27<age<=28', '28<age<=29', '29<age<=30', '30<age<=31', '31<age<=32', '32<age<=33', '33<age<=34', '34<age<=35', '35<age<=36', '36<age<=37', '37<age<=38', '38<age<=39', '39<age<=40', '40<age<=41', '41<age<=42', '42<age<=43', '43<age<=44', '44<age<=45', '45<age<=46', '46<age<=47', '47<age<=48', '48<age<=49', '49<age<=50', '50<age<=51', '51<age<=52', '52<age<=53', '53<age<=54', '54<age<=55', '55<age<=56', '56<age<=57', '57<age<=58', '58<age<=59', '59<age<=60', '60<age<=61', '61<age<=62', '62<age<=63', '63<age<=64', '64<age<=65', '65<age<=66', '66<age<=67', '67<age<=68', '68<age<=69', '69<age<=70', '70<age<=71', '71<age<=72', '72<age<=73', '73<age<=74', '74<age<=75', '75<age<=76', '76<age<=77', '77<age<=78', '78<age<=79', '79<age<=80', '80<age<=83', '83<age<=96', 'juv_fel_count<=0', '0<juv_fel_count<=1', '1<juv_fel_count<=2', '2<juv_fel_count<=3', '3<juv_fel_count<=4', '4<juv_fel_count<=5', '5<juv_fel_count<=6', '6<juv_fel_count<=8', '8<juv_fel_count<=9', '9<juv_fel_count<=10', '10<juv_fel_count<=20', 'juv_misd_count<=0', '0<juv_misd_count<=1', '1<juv_misd_count<=2', '2<juv_misd_count<=3', '3<juv_misd_count<=4', '4<juv_misd_count<=5', '5<juv_misd_count<=6', '6<juv_misd_count<=8', '8<juv_misd_count<=12', '12<juv_misd_count<=13', 'juvenile_crimes<=0', '0<juvenile_crimes<=1', '1<juvenile_crimes<=2', '2<juvenile_crimes<=3', '3<juvenile_crimes<=4', '4<juvenile_crimes<=5', '5<juvenile_crimes<=6', '6<juvenile_crimes<=7', '7<juvenile_crimes<=8', '8<juvenile_crimes<=9', '9<juvenile_crimes<=10', '10<juvenile_crimes<=11', '11<juvenile_crimes<=14', '14<juvenile_crimes<=20', 'priors_count<=0', '0<priors_count<=1', '1<priors_count<=2', '2<priors_count<=3', '3<priors_count<=4', '4<priors_count<=5', '5<priors_count<=6', '6<priors_count<=7', '7<priors_count<=8', '8<priors_count<=9', '9<priors_count<=10', '10<priors_count<=11', '11<priors_count<=12', '12<priors_count<=13', '13<priors_count<=14', '14<priors_count<=15', '15<priors_count<=16', '16<priors_count<=17', '17<priors_count<=18', '18<priors_count<=19', '19<priors_count<=20', '20<priors_count<=21', '21<priors_count<=22', '22<priors_count<=23', '23<priors_count<=24', '24<priors_count<=25', '25<priors_count<=26', '26<priors_count<=27', '27<priors_count<=28', '28<priors_count<=29', '29<priors_count<=30', '30<priors_count<=31', '31<priors_count<=33', '33<priors_count<=35', '35<priors_count<=36', '36<priors_count<=37', '37<priors_count<=38', 'current_charge_degree=felony<=0', '0<current_charge_degree=felony<=1']
header = pd.Index(["intercept"] + header)
header = header.astype("object")

list_of_weights = np.random.rand(3, 100)
# list_of_weights[list_of_weights < 0.7] = 0
list_of_weights[:, 0] = 0
for i in range(5, 10):
    list_of_weights[:, i] = np.random.random()
for i in range(80, 85):
    list_of_weights[:, i] = np.random.random()

def get_feature_thresholds(weights, columns):
    feature_thresholds = defaultdict(list)
    for col, weight in zip(columns, weights):
        match = re.search(r'([a-zA-Z]+)', col)
        if match:
            feature = match.group(1)
            threshold = re.findall(r'[\d.]+', col)
            if feature == 'juv':
                feature = 'juv_misd_count'
            if feature == 'juvenile':
                feature = 'juvenile_crimes'
            feature_thresholds[feature].append((list(map(float, threshold)), weight))
    return feature_thresholds

_, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
axs = axs.flatten()
ax_dict = {}
counter = 0
union_of_support_sets = defaultdict(int)
for i in tqdm(range(len(list_of_weights))):
    weights = list_of_weights[i, :]
    columns = header[np.nonzero(weights)[0]]
    weights = weights[np.nonzero(weights)[0]]
    for column in columns:
        union_of_support_sets[column] += 1

    feature_thresholds = get_feature_thresholds(weights, columns)
    for feature, thresholds_weights in feature_thresholds.items():
        if feature == 'sex' or feature == 'current':
            continue
        if feature not in ax_dict:
            ax = axs[counter]
            ax_dict[feature] = ax
            counter += 1
        else:
            ax = ax_dict[feature]
        thresholds, feature_weights = zip(*thresholds_weights)

        x_vals, y_vals = [], []
        x_vals.append(0)
        y_vals.append(feature_weights[0])
        for i in range(len(thresholds) - 1):
            x_vals.append(thresholds[i][0])
            y_vals.append(feature_weights[i])

        # For the last value of the interval, add it once more
        x_vals.append(thresholds[-1][0])
        y_vals.append(feature_weights[-1])

        ax.step(x_vals, y_vals, where="post", color='r', alpha=0.1)
        ax.set_ylabel("Predicted Logit")
        ax.set_title(feature)
plt.tight_layout()
plt.show()