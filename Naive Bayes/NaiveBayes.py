import scipy.stats
class NaiveBayesClassifier:

    def __init__(self):
        return

    def fit(self, X, y):
        self.X = X.copy()
        self.X["y"] = y

    def predict(self, test):
        preds = self.predict_proba(test)
        final_preds = []
        for i in range(len(test)):
            c = None
            max = 0
            for key in preds.keys():
                if preds[key][i] > max:
                    c = key
                    max = preds[key][i]
            final_preds.append(c)

        return final_preds

    def predict_proba(self, test):
        preds = []
        final = {}

        # looping over all labels
        for label in list(self.X["y"].unique()):
            probs = []

            # looping over row in the test set
            for row in range(len(test)):
                # num, deno = len(self.X[self.X["y"] == label]) / len(self.X), 1
                prob = len(self.X[self.X["y"] == label])/ len(self.X)
                # looping over cols in the test set
                for col in range(len(test.T)):
                    key = test.iloc[row, col]

                    if len(list(self.X.iloc[:, col].unique())) > 20:
                        prob *= scipy.stats.norm(self.X[self.X["y"] == label].iloc[:, col].mean(), self.X[self.X["y"] == label].iloc[:, col].std()).pdf(key)

                    else:
                        temp = len(self.X[self.X.iloc[:, col] == key])
                        deno_temp = temp / len(self.X)
                        # deno *= deno_temp
                        num_temp = len(self.X[(self.X.iloc[:, col] == key) & (self.X["y"] == label)]) / len(self.X[self.X["y"] == label])
                        # num *= num_temp
                        prob *= (num_temp / deno_temp)
                        
                probs.append(prob)
            final[label] = probs
        return final
