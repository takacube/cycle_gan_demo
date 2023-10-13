import datetime
import pickle
import pandas as pd
# import numpy as np
import luigi
from luigi.mock import MockTarget
import time

class GetData(luigi.Task):
    # get param
    input_path = luigi.Parameter()
    filename = luigi.Parameter()

    def requires(self):
        pass  # no requires

    def output(self):
        # return MockTarget(self.__class__.__name__, format=luigi.format.Nop)
        return luigi.LocalTarget('./tmp/GetData.pkl', format=luigi.format.Nop)

    def run(self):
        # get data
        # wait for 5 seconds
        time.sleep(5)
        inputfile_path = self.input_path + self.filename
        input_df = pd.read_csv(inputfile_path)

        # output input_df with the file path '.tmp/GetData.pkl'
        with self.output().open('w') as f:
            f.write(pickle.dumps(input_df, protocol=pickle.HIGHEST_PROTOCOL))


class PreprocessA(luigi.Task):
    def requires(self):
        return GetData()

    def output(self):
        # return MockTarget(self.__class__.__name__, format=luigi.format.Nop)
        return luigi.LocalTarget('./tmp/PreprocessA.pkl', format=luigi.format.Nop)

    def run(self):
        time.sleep(5)
        # load data which GetData output
        with self.input().open('r') as f:
            input_df: pd.DataFrame = pickle.load(f)

        # preprocess with input_df, and make result_df.
        result_df = input_df.drop(columns=['SepalLength'])

        # output result_df with the file path '.tmp/Preprocess.pkl'
        with self.output().open('w') as f:
            f.write(pickle.dumps(result_df, protocol=pickle.HIGHEST_PROTOCOL))


class PreprocessB(luigi.Task):
    def requires(self):
        return GetData()

    def output(self):
        # return MockTarget(self.__class__.__name__, format=luigi.format.Nop)
        return luigi.LocalTarget('./tmp/PreprocessB.pkl', format=luigi.format.Nop)

    def run(self):
        time.sleep(5)
        # load data which GetData output
        with self.input().open('r') as f:
            input_df: pd.DataFrame = pickle.load(f)

        # preprocess with input_df, and make result_df.
        result_df = input_df.drop(columns=['SepalWidth'])

        # output result_df with the file path '.tmp/Preprocess.pkl'
        with self.output().open('w') as f:
            f.write(pickle.dumps(result_df, protocol=pickle.HIGHEST_PROTOCOL))


class Sample(luigi.Task):
    # get param
    output_path = luigi.Parameter()
    # get datetime
    date = datetime.datetime.now()
    datetime = date.strftime("%Y%m%d_%H%M%S")
    datetime = str(datetime)

    def requires(self):
        return {'a': PreprocessA(), 'b': PreprocessB()}

    def output(self):
        pass

    def run(self):
        # input a
        with self.input()['a'].open('r') as f:
            input_a: pd.DataFrame = pickle.load(f)

        # input b
        with self.input()['b'].open('r') as f:
            input_b: pd.DataFrame = pickle.load(f)

        # process (marge)
        df = pd.concat([input_a, input_b], axis=1)

        # save df with the file path './output/result_%YYYY%MM%DD-HH%MM%SS.csv'
        outputfile_path = self.output_path + 'result_' + self.datetime + '.csv'
        df.to_csv(outputfile_path, index=False)

        # Delete files in folder = './tmp/*'
        # shutil.rmtree('./tmp/')


if __name__ == '__main__':
    luigi.configuration.LuigiConfigParser.add_config_path('./conf/config.cfg')
    # luigi.run(['Sample', '--local-scheduler'])
    luigi.run(['Sample', '--workers', '2', '--local-scheduler'])