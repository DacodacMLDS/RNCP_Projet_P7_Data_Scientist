#streamlit run dashboard.py

# Import all packages and libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
import pandas as pd
import numpy as np
import pickle
import math
import base64
from sklearn.ensemble import GradientBoostingClassifier
import streamlit as st
from collections import defaultdict
from PIL import Image 


# Fonctions
# Prédiction octroi de crédit
def predict_credit_default_client(model, id_client, data, seuil):
    ID = int(id_client)
    X = data[data['SK_ID_CURR'] == ID]
    X = X.drop(['SK_ID_CURR'], axis=1)
    proba_default = model.predict_proba(X)[:, 1]
    if proba_default >= seuil:
            prediction = "Client défaillant: prêt refusé"
    else:
            prediction = "Client non défaillant:  prêt accordé"
    return proba_default, prediction
    
# Chargement des tables du modèle
def load_data_model():
    # Ouverture du modèle, des données et prise en compte du threshold
    gbc = pickle.load(open('trained_gbc_model.pkl', 'rb'))

    #data = pd.read_csv('df_data_API.csv', sep=",")
    data = pd.read_csv('data_predict_api.csv', sep=",")
    all_id_client = data['SK_ID_CURR'].unique()

    # Données interprétables du client
    df_int = pd.read_csv('df_interprete_mod.csv')

    # import du modèle NearestNeighbors entrainé sur le trainset
    infile = open('NearestNeighborsModel.pkl', 'rb')
    nn = pickle.load(infile)
    infile.close()
    
    # import du Standard Scaler entrainé pour le NearestNeighbors
    infile = open('StandardScaler.pkl', 'rb')
    std = pickle.load(infile)
    infile.close()
    
    # import des données avec target pour modèle des plus proches voisins
    data_api_target = pd.read_csv('df_nn.csv')
    
    # import des données avec target sans standardisation
    data_api_with_target = pd.read_csv('data_api_target.csv')
    
    feature_important_data = ['SK_ID_CURR',
                          'CODE_GENDER',
                          'ANNUITY_CREDIT_PERCENT_INCOME',
                          'AMT_ANNUITY',
                          'AMT_INCOME_TOTAL',
                          'AGE',
                          'DAYS_EMPLOYED_PERCENT',
                          'CREDIT_REFUND_TIME']
    
    return gbc, data, all_id_client, df_int, feature_important_data, std, nn, data_api_target, data_api_with_target
    
# Vérification si le client est répertorié
def is_valid_client(id_client, all_client):
    if id_client not in all_client:
        return "ko"
    else:
        return "ok"
    
# Fonction pour la prédiction de crédit    
def predict_client(id_client):
    # Client non répertorié
    valid = is_valid_client(id_client, all_id_client)
    if valid == "ko":
        client_non_repertorie =  '<p class="p-style-red"><b>Ce client n\'est pas répertorié</b></p>'
        st.markdown(client_non_repertorie, unsafe_allow_html=True)    
        return "ko"

    else:
        st.markdown("##### Prédiction sur le client")  

    if valid == "ok":       
    
        proba_default, prediction = predict_credit_default_client(gbc, id_client, data, threshold)
        with st.container():
            desc_predict_client_col, predict_col = st.columns((2,2))
             
        with desc_predict_client_col:
            desc_predict = '<p class="p-style"><u>Probabilité de défaillance de paiement:</u>'
            st.markdown(desc_predict, unsafe_allow_html=True)
            
        with predict_col:
            # Prédiction
            if prediction == "Client non défaillant:  prêt accordé":
                predict = '<p class="p-style-green">{}%</p>'.format((proba_default[0]*100).round(2))
                st.markdown(predict, unsafe_allow_html=True)
            
            else:
                predict = '<p class="p-style-red">{}%</p>'.format((proba_default[0]*100).round(2))
                st.markdown(predict, unsafe_allow_html=True)
                
        with desc_predict_client_col:
            # Libellé prédiction
            if prediction == "Client non défaillant:  prêt accordé":
                lib_predict =  '<p class="p-style-green">{}</p>'.format(prediction)
                st.markdown(lib_predict, unsafe_allow_html=True)
                
            else:
                lib_predict =  '<p class="p-style-red">{}</p>'.format(prediction)
                st.markdown(lib_predict, unsafe_allow_html=True)
                
        desc_seuil = "On considère un client en défaillance de paiement si sa probabilité de défaillance est supérieure à 52 %"
        lib_desc_seuil =  '<p class="p-style-blue">{}</p>'.format(desc_seuil)
        st.markdown(lib_desc_seuil, unsafe_allow_html=True)        
                               
        return "ok"          
                               
    return "ko"            
                               
def infos_client(id_client, df_in):
    with st.container():
        grid_client = st.columns((10))
        
        # Client non répertorié
        valid = is_valid_client(id_client, all_id_client)
        if valid == "ko":
            return "ko"
                    
        else:
            st.markdown("##### Informations client")
            df_data = df_in.copy()
            df_data.drop('Cible', axis=1, inplace=True)
            df_client_int = df_data[df_data['Id client'] == id_client]
            grid_Options = config_aggrid(df_client_int)
            grid_response = AgGrid(
                                    df_client_int, 
                                    gridOptions=grid_Options,
                                    height=80, 
                                    width='100%',
                            )
                            
            return "ok"
                    
    return "ko" 


def client_sim(id_client, feature_important_data, select_chart):
    with st.container():
        grid_client_sim = st.columns((10))
        
        # Client non répertorié
        valid = is_valid_client(id_client, all_id_client)
        if valid == "ko":
            return "ko"
                    
        else:
            st.markdown("##### Profils de clients similaires")
            voisin = client_sim_voisins(feature_imp_data)
            if select_chart == "Tableau":
                grid_Options = config_aggrid_2(voisin)
                grid_response = AgGrid(
                                    voisin, 
                                    gridOptions=grid_Options,
                                    height=220, 
                                    width='100%',
                            )
            if select_chart == "Plot radar":
                radar_chart(voisin)
            
        return "ok"
                    
    return "ko" 

def client_graph_gen(id_client, df_in, var, type_graph):
    with st.container():
        st.markdown("##### Graphes généraux sur les clients")
        expl, grid_graph_gen_sim = st.columns((1,10))
        
        with expl:
            st.write("")
        with grid_graph_gen_sim:
            if type_graph == 'cible':
            
                if var == 'Age':
                    bar_plot_cible(df_in, 'pl Age client (ans)', 700, 700)
                elif var == '% Annuités/revenus':
                    bar_plot_cible(df_in, 'pl % annuités/revenus', 700, 700)
                else:
                    bar_plot_cible(df_in, var, 600, 600)                
                    
    return "ok" 
    
def client_graph_feat():
    with st.container():
        st.markdown("##### Graphe global sur l\'importance des variables")
        expl, grid_graph_feat = st.columns((1,10))
        
        with expl:
            st.write("")
        with grid_graph_feat:
            image = Image.open('feature_importance.png')
            st.image(image)
                    
    return "ok" 

def client_graph_det_feat(data, var):
    with st.container():    
        st.markdown("##### Graphes détaillés sur l\'importance des variables")
        expl, grid_graph_det_feat = st.columns((1,10))
        
        with expl:
            st.write("")
        with grid_graph_det_feat:
            
            if var == 'EXT_SOURCE_2':
                plot_feature_importance(data, 'TARGET', 'EXT_SOURCE_2','Distribution EXT_SOURCE_2', 'Cible', 'EXT_SOURCE_2')
                image = Image.open('EXT_SOURCE_2.png')
                st.image(image) 

            if var == 'EXT_SOURCE_3':
                plot_feature_importance(data, 'TARGET', 'EXT_SOURCE_3','Distribution EXT_SOURCE_3', 'Cible', 'EXT_SOURCE_3')
                image = Image.open('EXT_SOURCE_3.png')
                st.image(image)

            if var == 'DAYS_EMPLOYED':
                plot_feature_importance(data, 'TARGET', 'DAYS_EMPLOYED_PERC','Distribution DAYS_EMPLOYED_PERC', 'Cible', 'DAYS_EMPLOYED_PERC')
                image = Image.open('DAYS_EMPLOYED_PERC.png')
                st.image(image) 

            if var == 'CREDIT_REFUND_TIME':
                plot_feature_importance(data, 'TARGET', 'CREDIT_REFUND_TIME','Distribution CREDIT_REFUND_TIME', 'Cible', 'CREDIT_REFUND_TIME')
                image = Image.open('CREDIT_REFUND_TIME.png')
                st.image(image)

            if var == 'AGE':
                plot_feature_importance(data, 'TARGET', 'AGE','Distribution AGE', 'Cible', 'AGE')
                image = Image.open('AGE.png')
                st.image(image)          
    return "ok" 
    
# Classe pour configuration aggrid
class GridOptionsBuilder:
    """Builder for gridOptions dictionary"""

    def __init__(self):
        self.__grid_options = defaultdict(dict)
        self.sideBar = {}

    @staticmethod
    def from_dataframe(dataframe, **default_column_parameters):
        """
        Creates an instance and initilizes it from a dataframe.
        ColumnDefs are created based on dataframe columns and data types.
        Args:
            dataframe (pd.DataFrame): a pandas DataFrame.
        Returns:
            GridOptionsBuilder: The instance initialized from the dataframe definition.
        """

        # numpy types: 'biufcmMOSUV' https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
        type_mapper = {
            "b": ["textColumn"],
            "i": ["numericColumn", "numberColumnFilter"],
            "u": ["numericColumn", "numberColumnFilter"],
            "f": ["numericColumn", "numberColumnFilter"],
            "c": [],
            "m": ['timedeltaFormat'],
            "M": ["dateColumnFilter", "shortDateTimeFormat"],
            "O": [],
            "S": [],
            "U": [],
            "V": [],
        }

        gb = GridOptionsBuilder()
        gb.configure_default_column(**default_column_parameters)

        for col_name, col_type in zip(dataframe.columns, dataframe.dtypes):
            gb.configure_column(field=col_name, type=type_mapper.get(col_type.kind, []))

        return gb

    def configure_default_column(self, min_column_width=5, resizable=True, filterable=True, sorteable=True, editable=False, groupable=False, **other_default_column_properties):
        """Configure default column.
        Args:
            min_column_width (int, optional):
                Minimum column width. Defaults to 100.
            resizable (bool, optional):
                All columns will be resizable. Defaults to True.
            filterable (bool, optional):
                All columns will be filterable. Defaults to True.
            sorteable (bool, optional):
                All columns will be sorteable. Defaults to True.
            groupable (bool, optional):
                All columns will be groupable based on row values. Defaults to True.
            editable (bool, optional):
                All columns will be editable. Defaults to True.
            groupable (bool, optional):
                All columns will be groupable. Defaults to True.
            **other_default_column_properties:
                Key value pairs that will be merged to defaultColDef dict.
                Chech ag-grid documentation.
        """
        defaultColDef = {
            "minWidth": min_column_width,
            "editable": editable,
            "filter": filterable,
            "resizable": resizable,
            "sortable": sorteable,
        }
        if groupable:
            defaultColDef["enableRowGroup"] = groupable

        if other_default_column_properties:
            defaultColDef = {**defaultColDef, **other_default_column_properties}

        self.__grid_options["defaultColDef"] = defaultColDef

    def configure_auto_height(self, autoHeight=True):
        if autoHeight:
            self.configure_grid_options(domLayout='autoHeight')
        else:
            self.configure_grid_options(domLayout='normal')

    def configure_grid_options(self, **props):
        """Merges props to gridOptions
        Args:
            props (dict): props dicts will be merged to gridOptions root.
        """
        self.__grid_options.update(props)

    def configure_columns(self, column_names=[], **props):
        """Batch configures columns. Key-pair values from props dict will be merged
        to colDefs which field property is in column_names list.
        Args:
            column_names (list, optional):
                columns field properties. If any of colDefs mathces **props dict is merged.
                Defaults to [].
        """
        for k in self.__grid_options["columnDefs"]:
            if k in column_names:
                self.__grid_options["columnDefs"][k].update(props)

    def configure_column(self, field, header_name=None, **other_column_properties):
        """Configures an individual column
        check https://www.ag-grid.com/javascript-grid-column-properties/ for more information.
        Args:
            field (String): field name, usually equals the column header.
            header_name (String, optional): [description]. Defaults to None.
        """
        if not self.__grid_options.get("columnDefs", None):
            self.__grid_options["columnDefs"] = defaultdict(dict)

        colDef = {"headerName": header_name if header_name else field, "field": field}

        if other_column_properties:
            colDef = {**colDef, **other_column_properties}

        self.__grid_options["columnDefs"][field].update(colDef)

    def configure_side_bar(self, filters_panel=True, columns_panel=True, defaultToolPanel=""):
        """configures the side panel of ag-grid.
           Side panels are enterprise features, please check www.ag-grid.com
        Args:
            filters_panel (bool, optional):
                Enable filters side panel. Defaults to True.
            columns_panel (bool, optional):
                Enable columns side panel. Defaults to True.
            defaultToolPanel (str, optional): The default tool panel that should open when grid renders.
                                              Either "filters" or "columns".
                                              If value is blank, panel will start closed (default)
        """
        filter_panel = {
            "id": "filters",
            "labelDefault": "Filters",
            "labelKey": "filters",
            "iconKey": "filter",
            "toolPanel": "agFiltersToolPanel",
        }

        columns_panel = {
            "id": "columns",
            "labelDefault": "Columns",
            "labelKey": "columns",
            "iconKey": "columns",
            "toolPanel": "agColumnsToolPanel",
        }

        if filters_panel or columns_panel:
            sideBar = {"toolPanels": [], "defaultToolPanel": defaultToolPanel}

            if filters_panel:
                sideBar["toolPanels"].append(filter_panel)
            if columns_panel:
                sideBar["toolPanels"].append(columns_panel)

            self.__grid_options["sideBar"] = sideBar

    def configure_selection(
        self,
        selection_mode="single",
        use_checkbox=False,
        pre_selected_rows=None,
        rowMultiSelectWithClick=False,
        suppressRowDeselection=False,
        suppressRowClickSelection=False,
        groupSelectsChildren=True,
        groupSelectsFiltered=True,
    ):
        """Configure grid selection features
        Args:
            selection_mode (str, optional):
                Either 'single', 'multiple' or 'disabled'. Defaults to 'single'.
            pre_selected_rows (list, optional):
                Use list of dataframe row iloc index to set corresponding rows as selected state on load. Defaults to None.
            rowMultiSelectWithClick (bool, optional):
                If False user must hold shift to multiselect. Defaults to True if selection_mode is 'multiple'.
            suppressRowDeselection (bool, optional):
                Set to true to prevent rows from being deselected if you hold down Ctrl and click the row
                (i.e. once a row is selected, it remains selected until another row is selected in its place).
                By default the grid allows deselection of rows.
                Defaults to False.
            suppressRowClickSelection (bool, optional):
                Supress row selection by clicking. Usefull for checkbox selection for instance
                Defaults to False.
            groupSelectsChildren (bool, optional):
                When rows are grouped selecting a group select all children.
                Defaults to True.
            groupSelectsFiltered (bool, optional):
                When a group is selected filtered rows are also selected.
                Defaults to True.
        """
        if selection_mode == "disabled":
            self.__grid_options.pop("rowSelection", None)
            self.__grid_options.pop("rowMultiSelectWithClick", None)
            self.__grid_options.pop("suppressRowDeselection", None)
            self.__grid_options.pop("suppressRowClickSelection", None)
            self.__grid_options.pop("groupSelectsChildren", None)
            self.__grid_options.pop("groupSelectsFiltered", None)
            return

        if use_checkbox:
            suppressRowClickSelection = True
            first_key = next(iter(self.__grid_options["columnDefs"].keys()))
            self.__grid_options["columnDefs"][first_key]["checkboxSelection"] = True
        
        if pre_selected_rows:
            self.__grid_options['preSelectedRows'] = pre_selected_rows

        self.__grid_options["rowSelection"] = selection_mode
        self.__grid_options["rowMultiSelectWithClick"] = rowMultiSelectWithClick
        self.__grid_options["suppressRowDeselection"] = suppressRowDeselection
        self.__grid_options["suppressRowClickSelection"] = suppressRowClickSelection
        self.__grid_options["groupSelectsChildren"] = groupSelectsChildren
        self.__grid_options["groupSelectsFiltered"] = groupSelectsFiltered

    def configure_pagination(self, enabled=True, paginationAutoPageSize=True, paginationPageSize=10):
        """Configure grid's pagination features
        Args:
            enabled (bool, optional):
                Self explanatory. Defaults to True.
            paginationAutoPageSize (bool, optional):
                Calculates optimal pagination size based on grid Height. Defaults to True.
            paginationPageSize (int, optional):
                Forces page to have this number of rows per page. Defaults to 10.
        """
        if not enabled:
            self.__grid_options.pop("pagination", None)
            self.__grid_options.pop("paginationAutoPageSize", None)
            self.__grid_options.pop("paginationPageSize", None)
            return

        self.__grid_options["pagination"] = True
        if paginationAutoPageSize:
            self.__grid_options["paginationAutoPageSize"] = paginationAutoPageSize
        else:
            self.__grid_options["paginationPageSize"] = paginationPageSize

    def build(self):
        """Builds the gridOptions dictionary
        Returns:
            dict: Returns a dicionary containing the configured grid options
        """
        self.__grid_options["columnDefs"] = list(self.__grid_options["columnDefs"].values())

        return self.__grid_options
        
        
# Configuration aggrid stricte
def config_aggrid(df_in):
    # Infer basic colDefs from dataframe types
    gb = GridOptionsBuilder.from_dataframe(df_in)
    
    # Customize gridOptions
    gb.configure_default_column(value=True, enableRowGroup=False, editable=False, filterable=False, sorteable=False)
    
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()
    
    return gridOptions
    
# Configuration aggrid avec filtre et tri
def config_aggrid_2(df_in):
    # Infer basic colDefs from dataframe types
    gb = GridOptionsBuilder.from_dataframe(df_in)
    
    # Customize gridOptions
    gb.configure_default_column(value=True, enableRowGroup=False, editable=False, filterable=True, sorteable=True)
    gb.configure_pagination(enabled=True, paginationAutoPageSize=True, paginationPageSize=5)
    
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()
    
    return gridOptions    
   
   
# Clients similaires plus proches voisins
def client_sim_voisins(feature_imp_data):
    data_nn = data_api_target[data_api_target['SK_ID_CURR'] == idclient]
    client_list = std.transform(data_nn[feature_imp_data])  # standardisation
    distance, voisins = nn.kneighbors(client_list)
    voisins = voisins[0]  
    
    # Création d'un dataframe avec les voisins
    voisins_table = pd.DataFrame()
    for v in range(len(voisins)):
        voisins_table[v] = data_api_target[feature_imp_data].iloc[voisins[v]]   
    voisins_int = pd.DataFrame(index=range(len(voisins_table.transpose())),
                                       columns=df_int.columns)
                                       
    i = 0
    for id in voisins_table.transpose()['SK_ID_CURR']:
        voisins_int.iloc[i] = df_int[df_int['Id client'] == id]
        i += 1
           
    return voisins_int
    
def radar_chart(client):
    """Fonction qui trace le graphe radar du client comparé aux crédits accordés/refusés de ses proches voisins
    """

    def _invert(x, limits):
        """inverts a value x on a scale from
        limits[0] to limits[1]"""
        return limits[1] - (x - limits[0])

    def _scale_data(data, ranges):
        """scales data[1:] to ranges[0],
        inverts if the scale is reversed"""
        for d, (y1, y2) in zip(data, ranges):
            assert (y1 <= d <= y2) or (y2 <= d <= y1)

        x1, x2 = ranges[0]
        d = data[0]

        if x1 > x2:
            d = _invert(d, (x1, x2))
            x1, x2 = x2, x1

        sdata = [d]

        for d, (y1, y2) in zip(data[1:], ranges[1:]):
            if y1 > y2:
                d = _invert(d, (y1, y2))
                y1, y2 = y2, y1

            sdata.append((d - y1) / (y2 - y1) * (x2 - x1) + x1)

        return sdata

    class ComplexRadar():
        def __init__(self, fig, variables, ranges,
                     n_ordinate_levels=6):
            angles = np.arange(0, 360, (360. / len(variables)))

            axes = [fig.add_axes([0.1, 0.1, 0.9, 0.9], polar=True,
                                 label="axes{}".format(i))
                    for i in range(len(variables))]

            axes[0].set_thetagrids(angles, labels=[])

            for ax in axes[1:]:
                ax.patch.set_visible(False)
                ax.grid("off")
                ax.xaxis.set_visible(False)

            for i, ax in enumerate(axes):
                grid = np.linspace(*ranges[i],
                                   num=n_ordinate_levels)                 
                gridlabel = ["{}".format(round(x, 2))
                             for x in grid]
                if ranges[i][0] > ranges[i][1]:
                    grid = grid[::-1]  # hack to invert grid
                    # gridlabels aren't reversed
                gridlabel[0] = ""  # clean up origin
                ax.set_rgrids(grid, labels=gridlabel, angle=angles[i])
                ax.set_ylim(*ranges[i])

            ticks = angles
            ax.set_xticks(np.deg2rad(ticks))  # crée les axes suivant les angles, en radians
            ticklabels = variables
            ax.set_xticklabels(ticklabels, fontsize=10)  # définit les labels

            angles1 = np.linspace(0, 2 * np.pi, len(ax.get_xticklabels()) + 1)
            angles1[np.cos(angles1) < 0] = angles1[np.cos(angles1) < 0] + np.pi
            angles1 = np.rad2deg(angles1)
            labels = []
            for label, angle in zip(ax.get_xticklabels(), angles1):
                x, y = label.get_position()
                lab = ax.text(x, y - .5, label.get_text(), transform=label.get_transform(),
                              ha=label.get_ha(), va=label.get_va())
                lab.set_rotation(angle)
                lab.set_fontsize(16)
                lab.set_fontweight('bold')
                labels.append(lab)
            ax.set_xticklabels([])

            # variables for plotting
            self.angle = np.deg2rad(np.r_[angles, angles[0]])
            self.ranges = ranges
            self.ax = axes[0]

        def plot(self, data, *args, **kw):
            sdata = _scale_data(data, self.ranges)
            self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

        def fill(self, data, *args, **kw):
            sdata = _scale_data(data, self.ranges)
            self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw) 

    # data 
    variables = ("% annuités/revenus", 
                "Annuités",
                "Revenus globaux",
                "Age client (ans)",
                "% jours travaillés", 
                "Durée remb crédit (ans)"
                 )
    var_data = ["% annuités/revenus", 
                "Annuités",
                "Revenus globaux", 
                "Age client (ans)",
                "% jours travaillés", 
                "Durée remb crédit (ans)"
                 ]             
    data_ex = client.iloc[0][var_data]
    ranges = [(min(client["% annuités/revenus"]) - 5, max(client["% annuités/revenus"]) + 5),
              (min(client["Annuités"]) - 5000, max(client["Annuités"]) + 5000),
              (min(client["Revenus globaux"]) - 5000, max(client["Revenus globaux"]) + 5000),
              (min(client["Age client (ans)"]) - 5, max(client["Age client (ans)"]) + 5),
              (min(client["% jours travaillés"]) - 5, max(client["% jours travaillés"]) + 5),
              (min(client["Durée remb crédit (ans)"]) - 5, max(client["Durée remb crédit (ans)"]) + 5)
              ]
    
    # plotting
    fig1 = plt.figure(figsize=(7, 7))
    radar = ComplexRadar(fig1, variables, ranges)
    # Affichage des données du client
    radar.plot(data_ex, label='Notre client')
    radar.fill(data_ex, alpha=0.2)
    
    # Affichage de données du client similaires défaillants
    client_defaillant = client[client['Cible'] == 'Client défaillant']
    client_non_defaillant = client[client['Cible'] == 'Client non défaillant']
    
    data = {"% annuités/revenus" : [0.0],
            "Annuités": [0.0],
            "Revenus globaux" : [0.0],
            "Age client (ans)" : [0.0],
            "% jours travaillés" : [0.0],
            "Durée remb crédit (ans)" : [0.0]
           }
    
    client_non_defaillant_mean = pd.DataFrame(data)
    client_defaillant_mean = pd.DataFrame(data)
    
    client_non_defaillant_mean["% annuités/revenus"] = round(client_non_defaillant["% annuités/revenus"].mean(),1)
    client_non_defaillant_mean["Annuités"] = round(client_non_defaillant["Annuités"].mean(),1)
    client_non_defaillant_mean["Revenus globaux"] = round(client_non_defaillant["Revenus globaux"].mean(),1)
    client_non_defaillant_mean["Age client (ans)"] = round(client_non_defaillant["Age client (ans)"].mean(),1)
    client_non_defaillant_mean["% jours travaillés"] = round(client_non_defaillant["% jours travaillés"].mean(),1)
    client_non_defaillant_mean["Durée remb crédit (ans)"] = round(client_non_defaillant["Durée remb crédit (ans)"].mean(),1)
    data_non_client_def = client_non_defaillant_mean.iloc[0][var_data]
    
    client_defaillant_mean["% annuités/revenus"] = round(client_defaillant["% annuités/revenus"].mean(),1)
    client_defaillant_mean["Annuités"] = round(client_defaillant["Annuités"].mean(),1)
    client_defaillant_mean["Revenus globaux"] = round(client_defaillant["Revenus globaux"].mean(),1)
    client_defaillant_mean["Age client (ans)"] = round(client_defaillant["Age client (ans)"].mean(),1)
    client_defaillant_mean["% jours travaillés"] = round(client_defaillant["% jours travaillés"].mean(),1)
    client_defaillant_mean["Durée remb crédit (ans)"] = round(client_defaillant["Durée remb crédit (ans)"].mean(),1)
    data_client_def = client_defaillant_mean.iloc[0][var_data]
     
    radar.plot(data_non_client_def,
               label='Moyenne des clients similaires sans défaut de paiement',
               color='g')
    radar.plot(data_client_def,
               label='Moyenne des clients similaires avec défaut de paiement',
               color='r')           
        
    fig1.legend(bbox_to_anchor=(1.7, 1))

    #st.pyplot(fig1)
    st.pyplot(fig1)
    #st.plotly_chart(fig1)

def bar_plot_cible(df_in, var, width, height):
    df_g = df_in.groupby([var, 'Cible']).size().reset_index()
    df_g['percentage'] = df_in.groupby([var, 'Cible']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
    df_g.columns = [var, 'Cible', 'Nombre', 'Percentage']

    fig = px.bar(df_g, x=var, y='Nombre', color='Cible', text=df_g['Percentage'].apply(lambda x: '{0:1.2f}%'.format(x)))
    
    fig.update_layout(
        autosize=True,
        width=width,
        height=height
        )

    st.plotly_chart(fig)
    
def plot_feature_importance(df, col_x, col_y, title, x_label, y_label):
    fig1 = plt.figure(figsize=(5, 5))
    sns.boxplot(x=col_x,
                y=col_y, 
                data=df,
                width=0.5,                
               showmeans=True,
               showfliers=False,
               meanprops={"marker":"o",
                          "markerfacecolor":"red", 
                          "markeredgecolor":"red",
                          "markersize":"3"})
    plt.ylabel(y_label, size=10)
    plt.xlabel(x_label, size=10)
    plt.title(title, size=10)
    plt.xticks([0, 1], ['0 : Client non défaillant', '1 : Client défaillant'],
       rotation=0, fontsize=10)
    #plt.tick_params(axis='y', labelsize=3)
    plt.savefig(col_y + '.png')
    #st.pyplot(fig1)
    
st.set_page_config(layout='wide',
                   page_title="Dashboard interactif : décision accord/refus de crédit")
                   
threshold = 0.52


# Affichage des éléments dans la page                   
# Définition des styles pour les éléments de la page
st.markdown(
            '''
            <style>
            .p-style-green {
                font-size:20px;
                font-family:sans-serif;
                color:GREEN;
                vertical-align: text-top;
            }
            
            .p-style-red {
                font-size:20px;
                font-family:sans-serif;
                color:RED;
                vertical-align: top;
            }
            
            .p-style-blue {
                font-size:15px;
                font-family:sans-serif;
                color:BLUE;
                vertical-align: top;
            }
            
            .p-style {
                font-size:15px;
                font-family:sans-serif;
                vertical-align: top;
            }
            
            .p-style-sidebar {
                font-size:17px;
                font-family:sans-serif;
                font-weight: bold;
                vertical-align: top;
                text-decoration: underline;   
            }
            
            .p-style-sidebar-sub {
                font-size:15px;
                font-family:sans-serif;
                vertical-align: top;
                text-decoration: underline;   
            }
            
            </style>
            ''',
            unsafe_allow_html=True

        )
        
# Chargement des données et modèles
gbc, data, all_id_client, df_int, feature_imp_data, std, nn, data_api_target, data_api_with_target = load_data_model()

st.markdown('# Dashboard interactif Home Credit') 
st.markdown("###### Dashboard pour visualiser les informations sur le client et les profils similaires de client sur la défaillance de crédit")

lib_pred = '<p class="p-style-sidebar">Prédiction client:</p>'
st.sidebar.markdown(lib_pred, unsafe_allow_html=True)

# Saisie de l'identifiant client 
idclient = st.sidebar.text_input("Veuillez saisir un ID client :")

if idclient != "":
    idclient = int(idclient)
    
    # Prédiction client
    valid_predict = predict_client(idclient)
    if valid_predict == "ok":
        # Infos client
        valid_infos = infos_client(idclient, df_int)
        
        if valid_infos == "ok":
            # Affichage des clients avec le même profil
            lib_client_sim = '<p class="p-style-sidebar">Infos client et clients similaires:</p>'
            st.sidebar.markdown(lib_client_sim, unsafe_allow_html=True)
            
            clients_with_same_profile = st.sidebar.checkbox('Clients avec le même profil')
            if clients_with_same_profile:
                select_chart = st.sidebar.selectbox("Séléctionnez un type de représentation:", ['Tableau', 'Plot radar'])
                valid_sim = client_sim(idclient, feature_imp_data, select_chart)
                
            # Choix Graphes généraux et/ou Graphes sur importance caractéristique ou les 2 options
            lib_options_graph = '<p class="p-style-sidebar">Affichage des autres graphes:</p>'
            st.sidebar.markdown(lib_options_graph, unsafe_allow_html=True)
            
            selected_option = st.sidebar.multiselect("Sélectionnez une ou plusieurs options:",['Graphes généraux', 
                                                                                      'Graphes importance variables', 
                                                                                      'Tous les graphes'])

            if "Tous les graphes" in selected_option:
                selected_option = ['Graphes généraux', 'Graphes importance variables']
            
            if "Graphes généraux" in selected_option:    
                # Graphes généraux sur l'ensemble des clients
                lib_graph_gen = '<p class="p-style-sidebar-sub">Graphes généraux sur les clients:</p>'
                st.sidebar.markdown(lib_graph_gen, unsafe_allow_html=True)        
                gen_var_princ = st.sidebar.selectbox('Sélection de la variable du graphe:',
                                                    ('Genre (H/F)', 'Age', '% Annuités/revenus')) 
                                                    

                if gen_var_princ in ['Genre (H/F)', 'Age', '% Annuités/revenus']:
                    valid_graph_gen = client_graph_gen(idclient, df_int, gen_var_princ, 'cible')
                    

            if "Graphes importance variables" in selected_option:
                # Graphes sur l'importance des features     
                lib_graph_imp = '<p class="p-style-sidebar-sub">Graphes importance variables:</p>'
                st.sidebar.markdown(lib_graph_imp, unsafe_allow_html=True)    
                selected_option_feat = st.sidebar.multiselect('Sélection une ou plusieurs options:',
                                                      ('Graphe global', 'Détail de l\'importance des variables', 'Tous les graphes'))

                if "Tous les graphes" in selected_option_feat:
                    selected_option_feat = ['Graphe global', 'Détail de l\'importance des variables']    
        
                if 'Graphe global' in selected_option_feat:
                    valid_graph_feat = client_graph_feat()
                        
                if 'Détail de l\'importance des variables' in selected_option_feat:
                    gen_var_princ = st.sidebar.selectbox('Sélection de la variable du graphe:',
                                                    ('EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_EMPLOYED', 'CREDIT_REFUND_TIME', 'AGE')) 
                    valide_graph_det_feat = client_graph_det_feat(data_api_with_target, gen_var_princ)
                                