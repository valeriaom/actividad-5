"""Comparación reproducible de tres clasificadores con búsqueda y trazabilidad MLflow."""
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay, RocCurveDisplay)
from datos_prep import sha256

SEED=42

def design_models(frame):
    cols=[c for c in frame.columns if c!='y']
    categorical=frame[cols].select_dtypes(include=['object','string','category']).columns.tolist()
    numeric=[c for c in cols if c not in categorical]
    pre=ColumnTransformer([
        ('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),
        ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore'))]),categorical)
    ])
    models={
      'regresion_logistica':(LogisticRegression(max_iter=2000,random_state=SEED,class_weight='balanced'),{'model__C':[0.1,1,10]}),
      'random_forest':(RandomForestClassifier(random_state=SEED,class_weight='balanced',n_jobs=1),{'model__n_estimators':[100,200],'model__max_depth':[8,16],'model__min_samples_leaf':[1,5]}),
      'svm_lineal':(LinearSVC(dual=False,max_iter=5000,random_state=SEED,class_weight='balanced'),{'model__C':[0.1,1,10]})
    }
    return pre,models

def run_experiment(data_file, root, cv_folds=5):
    import mlflow
    import mlflow.sklearn
    root=Path(root).resolve(); data_file=Path(data_file).resolve()
    output=root/'resultados'; output.mkdir(parents=True,exist_ok=True)
    db_path=(root/'mlflow.db').resolve()
    tracking_uri=f'sqlite:///{db_path.as_posix()}'
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment('Actividad5_UCI_Bank_Marketing')
    df=pd.read_csv(data_file,dtype={'pdays':'string'})
    assert set(df['y'].unique())=={0,1},'Se requieren ambas clases'
    X=df.drop(columns='y');y=df['y']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
    pre,models=design_models(df)
    cv=StratifiedKFold(n_splits=cv_folds,shuffle=True,random_state=SEED)
    rows=[];scores_by_model={};fitted={}
    for name,(est,grid) in models.items():
        pipe=Pipeline([('prep',pre),('model',est)])
        gs=GridSearchCV(pipe,grid,scoring='roc_auc',cv=cv,n_jobs=-1,refit=True,return_train_score=True)
        start=time.perf_counter();gs.fit(X_train,y_train);elapsed=time.perf_counter()-start
        model=gs.best_estimator_;fitted[name]=model
        pred=model.predict(X_test)
        score=model.predict_proba(X_test)[:,1] if hasattr(model,'predict_proba') else model.decision_function(X_test)
        metrics={'accuracy':accuracy_score(y_test,pred),'precision':precision_score(y_test,pred,zero_division=0),
                 'recall':recall_score(y_test,pred,zero_division=0),'f1':f1_score(y_test,pred,zero_division=0),
                 'roc_auc':roc_auc_score(y_test,score),'cv_best_auc':gs.best_score_,'train_seconds':elapsed}
        # Los pliegues comparativos se mantienen independientes del conjunto de prueba.
        fold_scores=cross_val_score(model,X_train,y_train,cv=cv,scoring='roc_auc',n_jobs=-1)
        scores_by_model[name]=fold_scores
        metrics.update(cv_mean_auc=float(fold_scores.mean()),cv_std_auc=float(fold_scores.std(ddof=1)))
        runname=f'{name}_gridsearch'
        with mlflow.start_run(run_name=runname):
            mlflow.set_tags({'dataset':'UCI Bank Marketing additional-full','dataset_sha256':sha256(data_file),
                             'seed':str(SEED),'splits':'estratificado 80/20','excluded':'duration','cv_folds':str(cv_folds)})
            mlflow.log_params({'model_name':name,'scoring':'roc_auc','test_fraction':.2,**{k:str(v) for k,v in gs.best_params_.items()}})
            mlflow.log_metrics({k:float(v) for k,v in metrics.items()})
            mlflow.log_dict({'fold_auc':[float(v) for v in fold_scores]},'cv_fold_auc.json')
            mlflow.log_dict({'grid':grid,'best_params':gs.best_params_},'search_setup.json')
            cvpath=output/f'{name}_grid_results.csv';pd.DataFrame(gs.cv_results_).to_csv(cvpath,index=False)
            mlflow.log_artifact(str(cvpath),artifact_path='grid')
            fig,ax=plt.subplots(figsize=(6,5));ConfusionMatrixDisplay.from_predictions(y_test,pred,ax=ax,cmap='Blues',colorbar=False)
            ax.set_title(name);fig.tight_layout();img=output/f'{name}_confusion.png';fig.savefig(img,dpi=150);plt.close(fig);mlflow.log_artifact(str(img),artifact_path='graficas')
            fig,ax=plt.subplots(figsize=(6,5));RocCurveDisplay.from_predictions(y_test,score,ax=ax)
            fig.tight_layout();img=output/f'{name}_roc.png';fig.savefig(img,dpi=150);plt.close(fig);mlflow.log_artifact(str(img),artifact_path='graficas')
            mlflow.sklearn.log_model(
                sk_model=model,
                name='modelo',
                serialization_format='skops',
                skops_trusted_types=[
                    'numpy.dtype',
                    'sklearn.compose._column_transformer._RemainderColsList',
                    'sklearn.tree._tree.Tree',
                ],
            )
        rows.append({'modelo':name,**{k:float(v) for k,v in metrics.items()},'mejores_parametros':str(gs.best_params_)})
    results=pd.DataFrame(rows).sort_values('cv_mean_auc',ascending=False)
    results.to_csv(output/'comparacion_modelos.csv',index=False)
    pd.DataFrame(scores_by_model).to_csv(output/'metricas_por_fold.csv',index=False)
    # Prueba exploratoria: CV shares training observations, so folds aren't fully independent.
    top=results['modelo'].tolist()[:2]
    diff=scores_by_model[top[0]]-scores_by_model[top[1]]
    test={'comparacion':top,'media_diferencia_auc':float(diff.mean()),'advertencia':'Wilcoxon con 5 folds es exploratorio; dependencia entre folds y baja potencia.'}
    try:
        test['wilcoxon_p']=float(wilcoxon(diff).pvalue) if not np.allclose(diff,0) else 1.0
    except ValueError: test['wilcoxon_p']=None
    (output/'comparacion_estadistica.json').write_text(json.dumps(test,indent=2,ensure_ascii=False))
    return results,test,output

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    a=parser.parse_args();root=Path(a.root)
    print(run_experiment(root/'datos/datos_limp/bank_marketing_clean.csv',root)[0].to_string(index=False))
