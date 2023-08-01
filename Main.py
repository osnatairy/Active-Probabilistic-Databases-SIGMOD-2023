import concurrent
import os
import sys

from sklearn.ensemble import RandomForestClassifier
import copy
from ActiveConsentEvaluation import ActiveConsentEvaluation
from BooleanEvaluationModule import *
from KnownProbesRepository import KnownProbesRepository
from LearnerModule import LearnerModule, LC, LAL, Learn_Once, No_Learning, LAL_MODEL
from ProbeSelectorModule import SimpleMultiplicationWithIntentionalFading, UtilityOnly, UncertaintyOnly
from Scenario import Scenarios, NELL


def save_results_ALL(path,n_initial, Online,Offline,LAL_plus_ctu,LC_plus_Ctu,LAL_Only,LC_only,Random, iterations):
    new_data = {  "Online": Online, "Offline": Offline,"LAL_PLUS_CTU":LAL_plus_ctu,"LC_Plus_Ctu":LC_plus_Ctu,"LAL_Only":LAL_Only,"LC_Only":LC_only
                  ,"RANDOM":Random

                }
    df = pd.DataFrame(new_data, columns=["Online", "Offline","LAL_PLUS_CTU","LC_Plus_Ctu","LAL_Only","LC_Only","RANDOM"],index=['RO','General','Q_Value']*iterations)



    df.to_csv(path + "\Results_NELL_experiment_initials_{}_file.csv".format( n_initial))




def save_results(path,n_initial,BE_algo, Online,Offline):
    new_data = {  "Online": Online, "Offline": Offline,

                }
    df = pd.DataFrame(new_data, columns=["Online", "Offline"],index=[1,2,3,4,5])

    df.to_csv(path + "\Results_NELL_experiment_{}_initials_{}_file.csv".format( n_initial,BE_algo))


def save_results_EP_AND_GREEDY(path,n_initial,BE_algo, EP,Greedy):
    new_data = {'EP': EP, "Greedy":Greedy

                }
    df = pd.DataFrame(new_data, columns=['EP', "Greedy"],index=[0])

    df.to_csv(path + "\Results_NELL_experiment_{}_initials_{}_file_Basics.csv".format( n_initial,BE_algo))


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

def get_variables(ls_dnf):
    def foo(list_exps_dnf):
        variables = set()

        for exp in list_exps_dnf:
            variables = variables.union(set(list(exp.get_symbols())))

        return list(variables)

    k = 10
    chunks = chunkify(ls_dnf, k)
    threads_list = list()
    results = set()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(k):
            t = executor.submit(foo, chunks[i])
            threads_list.append(t)

        for t in threads_list:
            results = results.union(set(t.result()))

    results = list(set(results)).copy()
    # print(len(results))

    return results
def randomize_known_probes(scen, init_number):
    variables_indices = []
    variables=get_variables(scen.get_ls_dnf())
    for concept in variables:
        transaction_num = int(str(concept).split('v')[-1])
        index_X = scen.index_of(scen.get_X(), transaction_num)
        variables_indices.append(index_X)
    initial_indices=np.random.choice(list(set(range(2,scen.get_X().shape[0])).difference(set(variables_indices))), size=init_number, replace=False)
    return initial_indices



def Online_variant_RO(repo, scen):
    original_scen = copy.deepcopy(scen)
    original_repo = copy.deepcopy(repo)
    RO_Algorithm = RO()
    BooleanEvaluationModule_Online = BooleanEvaluationModule(BE_algorithm=RO_Algorithm)
    ProbeSelectorModule = UtilityOnly()
    learnerModule = LearnerModule(classifier=RandomForestClassifier(n_estimators=100),uncertainty_estimator=LC())
    architecture_1=ActiveConsentEvaluation(learnerModule,BooleanEvaluationModule_Online,ProbeSelectorModule)



    idx,truth_value=architecture_1.Evaluate_consent(repo,scen)
    print(idx)
    print(truth_value)
    scen = copy.deepcopy(original_scen)
    repo = copy.deepcopy(original_repo)
    return idx


def LC_Only(repo, scen):
    original_scen = copy.deepcopy(scen)
    original_repo = copy.deepcopy(repo)
    BooleanEvaluationModule_Online = BooleanEvaluationModule(BE_algorithm=RO())
    ProbeSelectorModule = UncertaintyOnly()
    learnerModule = LearnerModule(classifier=RandomForestClassifier(n_estimators=100),uncertainty_estimator=LC())
    architecture_1=ActiveConsentEvaluation(learnerModule,BooleanEvaluationModule_Online,ProbeSelectorModule)
    idx,truth_value=architecture_1.Evaluate_consent(repo,scen)
    print(idx)
    print(truth_value)
    scen = copy.deepcopy(original_scen)
    repo = copy.deepcopy(original_repo)
    return idx
def LAL_Only(repo, scen,lal_cls):
    original_scen = copy.deepcopy(scen)
    original_repo = copy.deepcopy(repo)
    BooleanEvaluationModule_Online = BooleanEvaluationModule(BE_algorithm=RO())
    ProbeSelectorModule = UncertaintyOnly()
    learnerModule = LearnerModule(classifier=RandomForestClassifier(n_estimators=100),uncertainty_estimator=LAL(lal_cls))
    architecture_1=ActiveConsentEvaluation(learnerModule,BooleanEvaluationModule_Online,ProbeSelectorModule)
    idx,truth_value=architecture_1.Evaluate_consent(repo,scen)
    print(idx)
    print(truth_value)
    scen = copy.deepcopy(original_scen)
    repo = copy.deepcopy(original_repo)
    return idx



def Online_variant(repo, scen, BE_algo):
    original_scen = copy.deepcopy(scen)
    original_repo = copy.deepcopy(repo)
    BooleanEvaluationModule_Online = BooleanEvaluationModule(BE_algorithm=BE_algo)
    ProbeSelectorModule = UtilityOnly()
    learnerModule = LearnerModule(classifier=RandomForestClassifier(n_estimators=100),uncertainty_estimator=LC())
    architecture_1=ActiveConsentEvaluation(learnerModule,BooleanEvaluationModule_Online,ProbeSelectorModule)



    idx,truth_value=architecture_1.Evaluate_consent(repo,scen)
    print(idx)
    print(truth_value)
    scen = copy.deepcopy(original_scen)
    repo = copy.deepcopy(original_repo)
    return idx


def variant_LC_plus_CtU(repo,scen,BE_algo):
    original_scen = copy.deepcopy(scen)
    original_repo = copy.deepcopy(repo)
    learnerModule = LearnerModule(classifier=RandomForestClassifier(n_estimators=100), uncertainty_estimator=LC())
    BooleanEvaluationModule_ = BooleanEvaluationModule(BE_algorithm=BE_algo)
    ProbeSelectorModule = SimpleMultiplicationWithIntentionalFading(scen.get_variables())
    architecture_1 = ActiveConsentEvaluation(learnerModule, BooleanEvaluationModule_, ProbeSelectorModule)
    idx, truth_value = architecture_1.Evaluate_consent(repo, scen)
    print(idx)
    print(truth_value)
    scen = copy.deepcopy(original_scen)
    repo = copy.deepcopy(original_repo)
    return idx



def variant_LAL_plus_CtU(repo,scen,BE_algo,lal_cls):
    original_scen = copy.deepcopy(scen)
    original_repo = copy.deepcopy(repo)
    learnerModule = LearnerModule(classifier=RandomForestClassifier(n_estimators=100), uncertainty_estimator=LAL(lal_cls))
    BooleanEvaluationModule_ = BooleanEvaluationModule(BE_algorithm=BE_algo)
    ProbeSelectorModule = SimpleMultiplicationWithIntentionalFading(scen.get_variables())
    architecture_1 = ActiveConsentEvaluation(learnerModule, BooleanEvaluationModule_, ProbeSelectorModule)
    idx, truth_value = architecture_1.Evaluate_consent(repo, scen)
    print(idx)
    print(truth_value)
    scen = copy.deepcopy(original_scen)
    repo = copy.deepcopy(original_repo)
    return idx



def Offline_Variant(repo, scen,BE_algo):
    original_scen = copy.deepcopy(scen)
    original_repo = copy.deepcopy(repo)
    learnerModule = Learn_Once(classifier=RandomForestClassifier(n_estimators=100))
    BooleanEvaluationModule_Q_Value = BooleanEvaluationModule(BE_algorithm=BE_algo)
    ProbeSelectorModule = UtilityOnly()
    architecture_1 = ActiveConsentEvaluation(learnerModule, BooleanEvaluationModule_Q_Value, ProbeSelectorModule)
    idx, truth_value = architecture_1.Evaluate_consent(repo, scen)
    scen = copy.deepcopy(original_scen)
    repo = copy.deepcopy(original_repo)
    return idx

def Random_Variant(repo,scen):

    original_scen = copy.deepcopy(scen)
    original_repo = copy.deepcopy(repo)
    Random_algorithm = Random_selection()
    learnerModule = No_Learning()
    BooleanEvaluationModule_Greedy = BooleanEvaluationModule(BE_algorithm=Random_algorithm)
    ProbeSelectorModule = UtilityOnly()
    architecture_1 = ActiveConsentEvaluation(learnerModule, BooleanEvaluationModule_Greedy, ProbeSelectorModule)
    idx, truth_value = architecture_1.Evaluate_consent(repo, scen)

    scen=copy.deepcopy(original_scen)
    repo=copy.deepcopy(original_repo)
    return idx

def Greedy_Variant(repo, scen):
    original_scen = copy.deepcopy(scen)
    original_repo = copy.deepcopy(repo)
    Greedy_algorithm=Greedy()
    learnerModule = No_Learning()
    BooleanEvaluationModule_Greedy = BooleanEvaluationModule(BE_algorithm=Greedy_algorithm)
    ProbeSelectorModule = UtilityOnly()
    architecture_1 = ActiveConsentEvaluation(learnerModule, BooleanEvaluationModule_Greedy, ProbeSelectorModule)
    idx, truth_value = architecture_1.Evaluate_consent(repo, scen)
    scen = copy.deepcopy(original_scen)
    repo = copy.deepcopy(original_repo)
    return idx
def EP_Variant(repo, scen,BE_algo):
    original_scen = copy.deepcopy(scen)
    original_repo = copy.deepcopy(repo)

    learnerModule = No_Learning()
    BooleanEvaluationModule_Greedy = BooleanEvaluationModule(BE_algorithm=BE_algo)
    ProbeSelectorModule = UtilityOnly()
    architecture_1 = ActiveConsentEvaluation(learnerModule, BooleanEvaluationModule_Greedy, ProbeSelectorModule)
    idx, truth_value = architecture_1.Evaluate_consent(repo, scen)
    print(idx)
    print(truth_value)
    scen = copy.deepcopy(original_scen)
    repo = copy.deepcopy(original_repo)
    return idx

if __name__ == '__main__':
    iterations = 1

    RO_Algorithm = RO()
    Q_Value_Algorithm = Q_Value()
    General_Algorithm = General()
    lal_cls=LAL_MODEL()


    algorithm = sys.argv[1]
    query = sys.argv[2]
    n_initial = int(sys.argv[3])

    BE_algo = RO_Algorithm
    if algorithm == "Q_Value_Algorithm":
        BE_algo = Q_Value_Algorithm
    elif algorithm == "General_Algorithm":
        BE_algo =General_Algorithm


    scen = Scenarios(NELL(query=query))

    path='NELL\{}_Results\{}_initials'.format(query,n_initial)
    ls_res_online=[]
    ls_res_offline=[]
    ls_res_Random=[]
    ls_LAL=[]
    ls_LC=[]
    ls_LAL_Only=[]
    ls_LC_Only=[]
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(0,iterations):
        initial_idx = randomize_known_probes(scen, n_initial)
        scen.experiment_init(initial_idx)

        repo = KnownProbesRepository(X_train=(scen.get_X())[initial_idx].astype(int),
                                     y_train=(scen.get_y())[initial_idx].astype(int))

        res_Random_Variant=int(Random_Variant(copy.deepcopy(repo),copy.deepcopy(scen)))
        ls_res_Random.append(int(res_Random_Variant))

        res_LAL_Only = LAL_Only(copy.deepcopy(repo), copy.deepcopy(scen),lal_cls)
        ls_LAL_Only.append(int(res_LAL_Only))

        res_LC_only = LC_Only(copy.deepcopy(repo),copy.deepcopy(scen))
        ls_LC_Only.append(int(res_LC_only))


        #run the choosen algorithm
        res_Online=Online_variant(copy.deepcopy(repo),copy.deepcopy(scen),BE_algo)
        ls_res_online.append(int(res_Online))

        res_Offline=Offline_Variant(copy.deepcopy(repo),copy.deepcopy(scen),BE_algo)
        ls_res_offline.append(int(res_Offline))

        res_LAL=variant_LAL_plus_CtU(copy.deepcopy(repo),copy.deepcopy(scen),BE_algo,lal_cls)
        ls_LAL.append(int(res_LAL))

        res_LC=variant_LC_plus_CtU(copy.deepcopy(repo),copy.deepcopy(scen), BE_algo)
        ls_LC.append(int(res_LC))


    #print results
    save_results_ALL(path,n_initial,ls_res_online,ls_res_offline,ls_LAL,ls_LC,ls_LAL_Only*3,ls_LC_Only*3,ls_res_Random*3,iterations)
