import os
import re
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import json
import numpy as np

split = {
    "train": {
                "MI_II": ["A.mat", "B.mat", "C.mat", "D.mat", "E.mat", "F.mat", "G.mat", "H.mat", "J.mat", "L.mat"], # OK
                "MI_LR": ["s09.mat", "s50.mat", "s25.mat", "s30.mat", "s39.mat", "s35.mat", "s18.mat", "s13.mat", "s23.mat", "s21.mat", "s03.mat", "s14.mat", "s47.mat", "s22.mat", "s32.mat", "s04.mat", "s46.mat", "s16.mat", "s31.mat", "s37.mat", "s29.mat", "s02.mat", "s52.mat", "s17.mat", "s51.mat", "s12.mat", "s10.mat", "s24.mat", "s11.mat", "s08.mat", "s36.mat", "s38.mat", "s40.mat", "s20.mat", "s19.mat", "s27.mat", "s33.mat", "s05.mat", "s44.mat", "s26.mat", "s01.mat", "s07.mat", "s15.mat", "s41.mat", "s28.mat", "s42.mat"], # OK
                #"MI_LR-s09", "MI_LR-s50", "MI_LR-s25", "MI_LR-s30", "MI_LR-s39", "MI_LR-s35", "MI_LR-s18", "MI_LR-s13", "MI_LR-s23", "MI_LR-s21", "MI_LR-s03", "MI_LR-s14", "MI_LR-s47", "MI_LR-s22", "MI_LR-s32", "MI_LR-s04", "MI_LR-s46", "MI_LR-s16", "MI_LR-s31", "MI_LR-s37", "MI_LR-s29", "MI_LR-s02", "MI_LR-s52", "MI_LR-s17", "MI_LR-s51", "MI_LR-s12", "MI_LR-s10", "MI_LR-s24", "MI_LR-s11", "MI_LR-s08", "MI_LR-s36", "MI_LR-s38", "MI_LR-s40", "MI_LR-s20", "MI_LR-s19", "MI_LR-s27", "MI_LR-s33", "MI_LR-s05", "MI_LR-s44", "MI_LR-s26", "MI_LR-s01", "MI_LR-s07", "MI_LR-s15", "MI_LR-s41", "MI_LR-s28", "MI_LR-s42",
                "MI_BBCI_IV_Berlin": ["BCICIV_calib_ds1a.mat", "BCICIV_calib_ds1b.mat", "BCICIV_calib_ds1c.mat", "BCICIV_calib_ds1d.mat", "BCICIV_calib_ds1e.mat", "BCICIV_eval_ds1a.mat", "BCICIV_eval_ds1b.mat", "BCICIV_eval_ds1c.mat", "BCICIV_eval_ds1d.mat", "BCICIV_eval_ds1e.mat", "BCICIV_eval_ds1f.mat", "BCICIV_eval_ds1g.mat"], # OK
                #"MI_BCI_IV_Berlin-BCICIV_calib_ds1a", "MI_BCI_IV_Berlin-BCICIV_calib_ds1b", "MI_BCI_IV_Berlin-BCICIV_calib_ds1c", "MI_BCI_IV_Berlin-BCICIV_calib_ds1d", "MI_BCI_IV_Berlin-BCICIV_calib_ds1e", "MI_BCI_IV_Berlin-BCICIV_eval_ds1a", "MI_BCI_IV_Berlin-BCICIV_eval_ds1b", "MI_BCI_IV_Berlin-BCICIV_eval_ds1c", "MI_BCI_IV_Berlin-BCICIV_eval_ds1d", "MI_BCI_IV_Berlin-BCICIV_eval_ds1e", "MI_BCI_IV_Berlin-BCICIV_eval_ds1f", "MI_BCI_IV_Berlin-BCICIV_eval_ds1g",
                "MI_BBCI_IV_Graz_a": ["A01T.gdf", "A02T.gdf", "A03T.gdf", "A04T.gdf", "A05T.gdf", "A06T.gdf", "A07T.gdf", "A08T.gdf", "A09T.gdf"],
                #"MI_BCI_IV_Graz_a-A01T", "MI_BCI_IV_Graz_a-A02T", "MI_BCI_IV_Graz_a-A03T", "MI_BCI_IV_Graz_a-A04T", "MI_BCI_IV_Graz_a-A05T", "MI_BCI_IV_Graz_a-A06T", "MI_BCI_IV_Graz_a-A07T", "MI_BCI_IV_Graz_a-A08T", "MI_BCI_IV_Graz_a-A09T",
                "MI_BBCI_IV_Graz_b": ["B0101", "B0102", "B0103", "B0104", "B0105", "B0201", "B0202", "B0203", "B0204", "B0205", "B0301", "B0302", "B0303", "B0304", "B0305", "B0401", "B0402", "B0403", "B0404", "B0405", "B0501", "B0502", "B0503", "B0504", "B0505", "B0601", "B0602", "B0603", "B060", "B0605", "B0701", "B0702", "B0703", "B0704", "B0705"],
                #"MI_BCI_IV_Graz_b-B0101", "MI_BCI_IV_Graz_b-B0102", "MI_BCI_IV_Graz_b-B0103", "MI_BCI_IV_Graz_b-B0104", "MI_BCI_IV_Graz_b-B0105", "MI_BCI_IV_Graz_b-B0201", "MI_BCI_IV_Graz_b-B0202", "MI_BCI_IV_Graz_b-B0203", "MI_BCI_IV_Graz_b-B0204", "MI_BCI_IV_Graz_b-B0205", "MI_BCI_IV_Graz_b-B0301", "MI_BCI_IV_Graz_b-B0302", "MI_BCI_IV_Graz_b-B0303", "MI_BCI_IV_Graz_b-B0304", "MI_BCI_IV_Graz_b-B0305", "MI_BCI_IV_Graz_b-B0401", "MI_BCI_IV_Graz_b-B0402", "MI_BCI_IV_Graz_b-B0403", "MI_BCI_IV_Graz_b-B0404", "MI_BCI_IV_Graz_b-B0405", "MI_BCI_IV_Graz_b-B0501", "MI_BCI_IV_Graz_b-B0502", "MI_BCI_IV_Graz_b-B0503", "MI_BCI_IV_Graz_b-B0504", "MI_BCI_IV_Graz_b-B0505", "MI_BCI_IV_Graz_b-B0601", "MI_BCI_IV_Graz_b-B0602", "MI_BCI_IV_Graz_b-B0603", "MI_BCI_IV_Graz_b-B0604", "MI_BCI_IV_Graz_b-B0605", "MI_BCI_IV_Graz_b-B0701", "MI_BCI_IV_Graz_b-B0702", "MI_BCI_IV_Graz_b-B0703", "MI_BCI_IV_Graz_b-B0704", "MI_BCI_IV_Graz_b-B0705",
                "MI_eegmmidb": ["S077", "S032", "S105", "S021", "S066", "S012", "S008", "S019", "S096", "S082", "S035", "S005", "S044", "S028", "S043", "S023", "S009", "S017", "S072", "S058", "S002", "S024", "S018", "S007", "S003", "S010", "S001", "S073", "S013", "S085", "S108", "S086", "S081", "S004", "S107", "S098", "S071", "S026", "S095", "S040", "S048", "S041", "S050", "S047", "S051", "S016", "S029", "S101", "S014", "S042", "S049", "S031", "S062", "S106", "S074", "S037", "S064", "S109", "S104", "S067", "S036", "S084", "S020", "S059", "S015", "S053", "S063", "S099", "S045", "S055", "S089", "S011", "S094", "S046", "S087", "S038", "S083", "S097", "S091", "S093", "S056", "S069", "S060", "S022"],
                #"MI_eegmmidb-S077", "MI_eegmmidb-S032", "MI_eegmmidb-S105", "MI_eegmmidb-S021", "MI_eegmmidb-S066", "MI_eegmmidb-S012", "MI_eegmmidb-S008", "MI_eegmmidb-S019", "MI_eegmmidb-S096", "MI_eegmmidb-S082", "MI_eegmmidb-S035", "MI_eegmmidb-S005", "MI_eegmmidb-S044", "MI_eegmmidb-S028", "MI_eegmmidb-S043", "MI_eegmmidb-S023", "MI_eegmmidb-S009", "MI_eegmmidb-S017", "MI_eegmmidb-S072", "MI_eegmmidb-S058", "MI_eegmmidb-S002", "MI_eegmmidb-S024", "MI_eegmmidb-S018", "MI_eegmmidb-S007", "MI_eegmmidb-S003", "MI_eegmmidb-S010", "MI_eegmmidb-S001", "MI_eegmmidb-S073", "MI_eegmmidb-S013", "MI_eegmmidb-S085", "MI_eegmmidb-S108", "MI_eegmmidb-S086", "MI_eegmmidb-S081", "MI_eegmmidb-S004", "MI_eegmmidb-S107", "MI_eegmmidb-S098", "MI_eegmmidb-S071", "MI_eegmmidb-S026", "MI_eegmmidb-S095", "MI_eegmmidb-S040", "MI_eegmmidb-S048", "MI_eegmmidb-S041", "MI_eegmmidb-S050", "MI_eegmmidb-S047", "MI_eegmmidb-S051", "MI_eegmmidb-S016", "MI_eegmmidb-S029", "MI_eegmmidb-S101", "MI_eegmmidb-S014", "MI_eegmmidb-S042", "MI_eegmmidb-S049", "MI_eegmmidb-S031", "MI_eegmmidb-S062", "MI_eegmmidb-S106", "MI_eegmmidb-S074", "MI_eegmmidb-S037", "MI_eegmmidb-S064", "MI_eegmmidb-S109", "MI_eegmmidb-S104", "MI_eegmmidb-S067", "MI_eegmmidb-S036", "MI_eegmmidb-S084", "MI_eegmmidb-S020", "MI_eegmmidb-S059", "MI_eegmmidb-S015", "MI_eegmmidb-S053", "MI_eegmmidb-S063", "MI_eegmmidb-S099", "MI_eegmmidb-S045", "MI_eegmmidb-S055", "MI_eegmmidb-S089", "MI_eegmmidb-S011", "MI_eegmmidb-S094", "MI_eegmmidb-S046", "MI_eegmmidb-S087", "MI_eegmmidb-S038", "MI_eegmmidb-S083", "MI_eegmmidb-S097", "MI_eegmmidb-S091", "MI_eegmmidb-S093", "MI_eegmmidb-S056", "MI_eegmmidb-S069", "MI_eegmmidb-S060", "MI_eegmmidb-S022",
                "MI_HGD": ["train"], # TO CHECK
                #"MI_HGD-1-train", "MI_HGD-2-train", "MI_HGD-3-train", "MI_HGD-4-train", "MI_HGD-5-train", "MI_HGD-6-train", "MI_HGD-7-train", "MI_HGD-8-train", "MI_HGD-9-train", "MI_HGD-10-train", "MI_HGD-11-train", "MI_HGD-12-train", "MI_HGD-13-train", "MI_HGD-14-train",
                "MI_Two": ["1T.mat", "2T.mat", "3T.mat", "4T.mat", "5T.mat", "6T.mat", "7T.mat", "8T.mat", "9T.mat", "10T.mat", "11T.mat", "12T.mat", "13T.mat", "14T.mat"],
                #"MI_Two-1T", "MI_Two-2T", "MI_Two-3T", "MI_Two-4T", "MI_Two-5T", "MI_Two-6T", "MI_Two-7T", "MI_Two-8T", "MI_Two-9T", "MI_Two-10T", "MI_Two-11T", "MI_Two-12T", "MI_Two-13T", "MI_Two-14T",
                
                "MI_SCP": ["SubjectA-", "SubjectB-", "SubjectC-", "SubjectD-", "SubjectE-", "SubjectG-", "SubjectH-", "SubjectK-", "SubjectL-", "SubjectM-"],
                "MI_GVH": ["G1.mat", "G2.mat", "G3.mat", "G4.mat", "G5.mat", "G6.mat", "G7.mat", "G8.mat", "G9.mat", "G10.mat", "H1.mat", "H2.mat", "H3.mat", "H4.mat", "H5.mat", "H6.mat", "H7.mat", "H8.mat", "H9.mat", "H10.mat", "V1.mat", "V2.mat", "V3.mat", "V4.mat", "V5.mat", "V6.mat", "V7.mat", "V8.mat", "V9.mat", "V10.mat"],
                #"MI_GVH-G01", "MI_GVH-G02", "MI_GVH-G03", "MI_GVH-G04", "MI_GVH-G05", "MI_GVH-G06", "MI_GVH-G07", "MI_GVH-G08", "MI_GVH-G09", "MI_GVH-G10", "MI_GVH-H01", "MI_GVH-H02", "MI_GVH-H03", "MI_GVH-H04", "MI_GVH-H05", "MI_GVH-H06", "MI_GVH-H07", "MI_GVH-H08", "MI_GVH-H09", "MI_GVH-H10", "MI_GVH-V01", "MI_GVH-V02", "MI_GVH-V03", "MI_GVH-V04", "MI_GVH-V05", "MI_GVH-V06", "MI_GVH-V07", "MI_GVH-V08", "MI_GVH-V09", "MI_GVH-V10",
                "MI_GAL": ["_P1_", "_P2_", "_P3_", "_P4_", "_P5_", "_P6_", "_P7_", "_P8_", "_P9_"],
                "MI_ULM": ["_subject1_", "_subject2_", "_subject3_", "_subject4_", "_subject5_", "_subject6_", "_subject7_", "_subject8_", "_subject9_", "_subject10_", "_subject11_", "_subject12_"],
                "MI_SCI": ["P1Run", "P2", "P3", "P4", "P5", "P6", "P0", "P8"],
                "ErrP_BCI_NER": ["S2", "S6", "S7", "S11", "S12", "S13", "S14", "S16", "S17", "S18", "S20", "S21", "S22", "S23", "S24", "S26"],
                "ErrP_MERP": ["Subject01", "Subject02", "Subject03", "Subject04", "Subject05"],
                "ERP_ANA": ["Subject1", "Subject2", "Sbject3", "Subject4", "Subject5"],
                "ERP_BBI": ['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5', 'Subject6', 'Subject7', 'Subject8', 'Subject9', 'Subject10', 'Subject11', 'Subject12', 'Subject13', 'Subject14', 'Subject15', 'Subject16', 'Subject17', 'Subject18', 'Subject19', 'Subject20'],
                "ERP_BICD": ['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5', 'Subject6', 'Subject7', 'Subject8', 'Subject9', 'Subject10', 'Subject11', 'Subject12', 'Subject13', 'Subject14', 'Subject15', 'Subject16', 'Subject17', 'Subject18', 'Subject19', 'Subject20', 'Subject21', 'Subject22', 'Subject23', 'Subject24', 'Subject25', 'Subject26', 'Subject27', 'Subject28', 'Subject29', 'Subject30', 'Subject31', 'Subject32', 'Subject33', 'Subject34', 'Subject35', 'Subject36', 'Subject37', 'Subject38', 'Subject39', 'Subject40', 'Subject41', 'Subject42', 'Subject43', 'Subject44', 'Subject45', 'Subject46', 'Subject47', 'Subject48', 'Subject49', 'Subject50', 'Subject51', 'Subject52', 'Subject53', 'Subject54', 'Subject55', 'Subject56', 'Subject57'],
                "RS_ALPHA": ['subject_0', 'subject_1', 'subject_2', 'subject_3', 'subject_4', 'subject_5', 'subject_6', 'subject_7', 'subject_8', 'subject_9', 'subject_10', 'subject_11', 'subject_12', 'subject_13', 'subject_14', 'subject_15', 'subject_16', 'subject_17'],
                "RS_SPIS":  ["S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09"]
            },
    "test":  {
                "MI_Limb": ['wx.mat', 'yyx.mat', 'zd.mat', 'lnn.mat', 'cl.mat', 'ls.mat', 'ry.mat', 'wcf.mat', 'cyy.mat', 'kyf.mat'], # OK
                "MI_LR": ["s45.mat", "s49.mat", "s48.mat", "s34.mat", "s06.mat", "s43.mat"],
                #"MI_LR-s45", "MI_LR-s49", "MI_LR-s48", "MI_LR-s34", "MI_LR-s06", "MI_LR-s43",
                "MI_BBCI_IV_Berlin": ["BCICIV_calib_ds1f.mat", "BCICIV_calib_ds1g.mat"],
                #"MI_BCI_IV_Berlin-BCICIV_calib_ds1f", "MI_BCI_IV_Berlin-BCICIV_calib_ds1g",
                "MI_BBCI_IV_Graz_a": ["A01E.gdf", "A02E.gdf", "A03E.gdf", "A04E.gdf", "A05E.gdf", "A06E.gdf", "A07E.gdf", "A08E.gdf", "A09E.gdf"],
                #"MI_BCI_IV_Graz_a-A01E", "MI_BCI_IV_Graz_a-A02E", "MI_BCI_IV_Graz_a-A03E", "MI_BCI_IV_Graz_a-A04E", "MI_BCI_IV_Graz_a-A05E", "MI_BCI_IV_Graz_a-A06E", "MI_BCI_IV_Graz_a-A07E", "MI_BCI_IV_Graz_a-A08E", "MI_BCI_IV_Graz_a-A09E",
                "MI_BBCI_IV_Graz_b": ["B0801", "B0802", "B0803", "B0804", "B0805", "B0901", "B0902", "B0903", "B0904", "B0905"],
                #"MI_BCI_IV_Graz_b-B0801", "MI_BCI_IV_Graz_b-B0802", "MI_BCI_IV_Graz_b-B0803", "MI_BCI_IV_Graz_b-B0804", "MI_BCI_IV_Graz_b-B0805", "MI_BCI_IV_Graz_b-B0901", "MI_BCI_IV_Graz_b-B0902", "MI_BCI_IV_Graz_b-B0903", "MI_BCI_IV_Graz_b-B0904", "MI_BCI_IV_Graz_b-B0905",
                "MI_eegmmidb": ["S025", "S030", "S068", "S079", "S057", "S054", "S006", "S065", "S080", "S039", "S061", "S075", "S033", "S078", "S052", "S102", "S070", "S076", "S027", "S103", "S034"],
                #"MI_eegmmidb-S025", "MI_eegmmidb-S030", "MI_eegmmidb-S068", "MI_eegmmidb-S079", "MI_eegmmidb-S057", "MI_eegmmidb-S054", "MI_eegmmidb-S006", "MI_eegmmidb-S065", "MI_eegmmidb-S080", "MI_eegmmidb-S039", "MI_eegmmidb-S061", "MI_eegmmidb-S075", "MI_eegmmidb-S033", "MI_eegmmidb-S078", "MI_eegmmidb-S052", "MI_eegmmidb-S102", "MI_eegmmidb-S070", "MI_eegmmidb-S076", "MI_eegmmidb-S027", "MI_eegmmidb-S103", "MI_eegmmidb-S034",
                "MI_HGD": ["test"], # TO CHECK
                #"MI_HGD-1-test", "MI_HGD-2-test", "MI_HGD-3-test", "MI_HGD-4-test", "MI_HGD-5-test", "MI_HGD-6-test", "MI_HGD-7-test", "MI_HGD-8-test", "MI_HGD-9-test", "MI_HGD-10-test", "MI_HGD-11-test", "MI_HGD-12-test", "MI_HGD-13-test", "MI_HGD-14-test",
                "MI_Two": ["1E.mat", "2E.mat", "3E.mat", "4E.mat", "5E.mat", "6E.mat", "7E.mat", "8E.mat", "9E.mat", "10E.mat", "11E.mat", "12E.mat", "13E.mat", "14E.mat"],
                #"MI_Two-1E", "MI_Two-2E", "MI_Two-3E", "MI_Two-4E", "MI_Two-5E", "MI_Two-6E", "MI_Two-7E", "MI_Two-8E", "MI_Two-9E", "MI_Two-10E", "MI_Two-11E", "MI_Two-12E", "MI_Two-13E", "MI_Two-14E",
                "MI_SCP": ["SubjectF-", "SubjectI-", "SubjectJ-"],
                #"SubjectF-",  "SubjectI-", "SubjectJ-",
                "MI_GVH": ["G11.mat", "G12.mat", "G13.mat", "G14.mat", "G15.mat", "H11.mat", "H12.mat", "H13.mat", "H14.mat", "H15.mat",  "V11.mat", "V12.mat", "V13.mat", "V14.mat", "V15.mat"],
                #"MI_GVH-G11", "MI_GVH-G12", "MI_GVH-G13", "MI_GVH-G14", "MI_GVH-G15", "MI_GVH-H11", "MI_GVH-H12", "MI_GVH-H13", "MI_GVH-H14", "MI_GVH-H15",  "MI_GVH-V11", "MI_GVH-V12", "MI_GVH-V13", "MI_GVH-V14", "MI_GVH-V15",
                "MI_GAL": ["_P10_", "_P11_", "_P12_"],
                "MI_ULM": ["_subject13_", "_subject14_", "_subject15_"],
                "MI_SCI": ["P9", "P10Run"],
                "ErrP_BCI_NER": ["S1", "S3", "S4", "S5", "S8", "S9", "S10", "S15", "S19", "S25"],
                "ErrP_MERP": ["Subject06"],
                "ERP_ANA": ["subject06", "subject07"],
                "ERP_BBI": ["Subject21", "Subject22", "Subject23", "Subject24", "Subject25"],
                "ERP_BICD": ['Subject58', 'Subject59', 'Subject60', 'Subject61', 'Subject62', 'Subject63', 'Subject64'],
                "ERP_BICF": ['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5', 'Subject6', 'Subject7', 'Subject8', 'Subject9', 'Subject10', 'Subject11', 'Subject12', 'Subject13', 'Subject14', 'Subject15', 'Subject16', 'Subject17', 'Subject18', 'Subject19', 'Subject20', 'Subject21', 'Subject22', 'Subject23', 'Subject24', 'Subject25', 'Subject26', 'Subject27', 'Subject28', 'Subject29', 'Subject30', 'Subject31', 'Subject32', 'Subject33', 'Subject34', 'Subject35', 'Subject36', 'Subject37', 'Subject38', 'Subject39', 'Subject40', 'Subject41', 'Subject42', 'Subject43'],
                "RS_ALPHA": ['subject_18', 'subject_19', 'subject_20'],
                "RS_SPIS":  ["S10", "S11"]
    }
}


map_labels = {
    "MI_Limb": {
        'Left_Hand': "left hand imagined movement",
        'Right_Hand': "right hand imagined movement",
        'Feet': "left foot or right foot imagined movement",
        'Both_Hands': "both hand imagined movement",
        'Left_Hand_Right_Feet': "na", # We do not want these confusing trials for our benchmark
        'Right_Hand_Left_Feet': "na",
        'Rest': "rest"
    }, 
    "MI_BBCI_IV_Berlin": {
    # MI_BCI_IV_Berlin
        'imagines moving left hand': "left hand imagined movement",
        'imagines moving right hand': "right hand imagined movement",
        'imagines moving foot (side chosen by the subject; optionally also both feet)': "left foot or right foot imagined movement"
    },
    "MI_BBCI_IV_Graz_a": {
        # MI_BCI_IV_a
        '(1)Start of a new run (2)keep opening eyes': 'eye no movement eye open',
        '(1)Start of a new run (2)keep closing eyes': 'eye no movement eye closed',
        '(1)Start of a new run (2)move eyes': 'eye movement eye saccade',
        '(1)keep closing eyes (2)Rejected trial': 'eye no movement eye closed',
        'Rejected trial': 'Rejected trial',
        'move eyes': 'eye movement eye saccade',
        'keep closing eyes': 'eye no movement eye closed',
        'Start of a new run': 'new run',
        'Start of a trial': 'new trial',
        'imagines moving left hand': 'left hand imagined movement',
        'imagines moving right hand': 'right hand imagined movement',
        'imagines moving both feet': 'left foot or right foot imagined movement',
        'imagines moving tongue': 'tongue imagined movement'
    },
    "MI_BBCI_IV_Graz_b": {
        # MI_BCI_IV_b
        'open eyes': 'eye no movement eye open',
        'close eyes': 'eye no movement eye closed',
        'Start of a trial': 'new trial',
        'imagine moving left hand': 'left hand imagined movement',
        'imagine moving right hand': 'right hand imagined movement',
        'BCI feedback (continuous)': 'BCI feedback',
        'Unknown cue': 'unknown',
        'Rejected trial': 'rejected trial',
        'move eyes horizontally': 'eye movement eye saccade horizontal',
        'move eyes vertically': 'eye movement eye saccade vertical',
        'rotate eyes': 'eye movement eye saccade rotation',
        'blink eyes': 'eye movement eye blink',
        'Start of a new run': 'new run'
    },
    "MI_II": {
        # MI_II
        'Participant imagines mental word association': "imagined mental word association",
        'Participant imagines mental subtraction': "imagined mental subtraction",
        'Participant imagines spatial navigation': "imagined spatial navigation",
        'Participant imagines moving right hand': "right hand imagined movement",
        'Participant is relaxing and fixating the cross on the screen to avoid eye movements': "eye no movement fixation",
        'Participant imagines moving both feet': "left foot or right foot imagined movement",
    },
    "MI_LR": {
        # MI_LR
        'eye blinking': 'eye movement eye blink',
        'eyeball movement up/down': 'eye movement eye saccade vertical',
        'eyeball movement left/right': 'eye movement eye saccade horizontal',
        'head movement': 'head movement',
        'jaw clenching': 'jaw clenching',
        'moving left hand': "left hand movement",
        'imagines moving left hand': "left hand imagined movement",
        'moving right hand': "right hand movement",
        'imagines moving right hand': "right hand imagined movement",
    },
    "MI_eegmmidb": {
        # eegmmidb
        "Rest": "rest",
        'open and close the left fist': 'left hand movement left fist open and close',
        'imagine opening and closing the left fist': 'left hand imagined movement left fist imagined open and close',
        'open and close both fists': 'both hand movement both fist open and close',
        'imagine opening and closing both fists': 'both hand imagined movement both fist imagined open and close',
        'According to dataset description, there might be a fault for event description in this record': 'faulty event description',
        'open and close the right fist': 'right hand movement right fist open and close',
        'imagine opening and closing the right fist': 'right hand imagined movement right fist imagined open and close',
        'open and close both feet': 'left foot or right foot movement open and close',
        'imagine opening and closing both feet': 'left foot or right foot imagined movement open and close',
    },
    "MI_HGD": {
        # HGD
        'Participant moves feet': 'left foot or right foot movement clenching toes',
        'Participant moves left hand': 'left hand movement sequential finger-tapping',
        'Participant takes a rest': 'rest',
        'Participant moves right hand': 'right hand movement sequential finger-tapping'
    },
    "MI_Two": {
        # Two
        'right Hand': 'right hand imagined movement',
        'feet': 'left foot or right foot imagined movement'
    },
    "MI_SCP": {
        # SCP
        # 5f
        'imagines moving thumb': 'thumb imagined movement',
        'imagines moving index finger': 'index finger imagined movement',
        'imagines moving middle finger': 'middle finger imagined movement',
        'imagines moving ring finger': 'ring finger imagined movement',
        'imagines moving pinkie finger': 'pinkie finger imagined movement',
        'Participant is in initial relaxation period': 'relax',
        'Participant is in rest break period': 'rest',
        'Experiment ends': 'end experiment',
        'Unknown': 'unknown',
        # CLA
        'imagines moving left hand': 'left hand imagined movement',
        'imagines moving right hand': 'right hand imagined movement',
        'Participant is in passive state': 'passive state',
        'imagines moving left leg': 'left foot imagined movement',
        'imagines moving tongue': 'tongue imagined movement',
        'imagines moving right leg': 'right foot imagined movement',
        'Participant is in rest break period': 'rest',
        'Experiment ends': 'end experiment',
        'Participant is in initial relaxation period': 'relax',
        'Unknown': 'unknown',
        # FreeForm
        'presses keys by using left hand': 'left hand movement pressing key',
        'presses keys by using right hand': 'right hand movement pressing key',
        'Participant is in rest break period': 'rest',
        'Experiment ends': 'end experiment',
        'Participant is in initial relaxation period': 'relax',
        'Unknown': 'unknown',
        # NoMT
        'watches screen (signal indicates left hand)': 'no movement',
        'watches screen (signal indicates right hand)': 'no movement',
        'Participant is in passive state': 'passive state',
        'watches screen (signal indicates left leg)': 'no movement',
        'watches screen (signal indicates tongue)': 'no movement',
        'watches screen (signal indicates right leg)': 'no movement',
        'Participant is in rest break period': 'rest',
        'Experiment ends': 'end experiment',
        'Participant is in initial relaxation period': 'relax',
        'Unknown': 'unknown',
    },
    "MI_GVH": {
        # GVH
        'Participant moves right hand towards the object to be ready for performing palmar grasp': "right hand movement towards object",
        'Participant performs palmar grasp': "right hand movement palmar grasp",
        'Participant is in inter-trial interval': "rest",
        'Participant moves right hand towards the object to be ready for performing lateral grasp': "right hand movement towards object",
        'Participant performs lateral grasp': "right hand movement lateral grasp",
        'Participant takes rest': "rest",
        'Participant moves eyes vertically': "eye movement eye saccade vertical",
        'Participant moves eyes horizontally': "eye movement eye saccade horizontal",
        'Participant blinks eyes': "eye movement eye blink",
        "Unknown Event": "unknown",
    },
    "MI_GAL": {
        # GAL
        'Unknown': 'unknown',
        'starts to move hands to reach for the object': 'hand movement towards object',
        'touches the object': 'hand movement touching object',
        'grips object with forefinger and thumb and gets ready to lift it': 'hand movement gripping object',
        'lifts off the object and holds it stably': 'hand movement lifting object',
        'lowers the object and replace it': 'hand movement lowering object',
        'releases forefinger and thumb from the object': 'hand movement releasing object',
    },
    "MI_ULM": {
        # ULM (except s1 all right handed)
        'Participant performs elbow flexion': 'hand movement elbow flexion',
        'Participant imagines performing elbow flexion': 'hand movement imagined elbow flexion ',
        'Participant performs elbow extension': 'hand movement elbow extension',
        'Participant imagines performing elbow extension': 'hand movement imagined elbow extension ',
        'Participant performs supination': 'hand movement supination',
        'Participant imagines performing supination': 'hand movement imagined supination ',
        'Participant performs pronation': 'hand movement pronation',
        'Participant imagines performing pronation': 'hand movement imagined pronation ',
        'Participant performs hand closing': 'hand movement hand closing',
        'Participant imagines performing hand closing': 'hand imagined movement imagined hand closing',
        'Participant performs hand opening': 'hand movement hand opening',
        'Participant imagines performing hand opening': 'hand imagined movement imagined hand opening',
        'Participant moves back to the starting position and takes a break': 'rest',
    },
    "MI_SCI": {
        # MI_SCI
        'There is a beep and a cross as well as class cue pops up on the computer screen': 'cross and class cue',
        'Class cue disappears and participant starts to attempt multiple self-paced movements of palmar grasp': 'hand imagined movement hand attempted movement imagined palmar grasp',
        'Class cue disappears and participant starts to attempt multiple self-paced movements of hand opening': 'hand imagined movement hand attempted movement imagined hand opening',
        'Class cue disappears and participant takes a rest': 'rest',
        'Participant is in resting state': 'rest',
        'There is a beep and a cross pops up on the computer screen': 'rest',
        'There is a Ready cue on the screen': 'rest',
        'There is a Go cue on the screen and participant starts to perform palmar grasp': 'hand imagined movement hand attempted movement imagined palmar grasp',
        'There is a Go cue on the screen and participant starts to perform hand opening': 'hand imagined movement hand attempted movement imagined hand opening',
        'Participant takes a break': 'rest',
        'Participant moves and blinks eyes': 'eye movement eye saccade and eye blink',
        'Participant is in resting state': 'rest',
        'Participant performs supination': 'hand imagined movement hand attempted movement imagined supination',
        'Participant performs pronation': 'hand imagined movement hand attempted movement imagined pronation',
        'Participant performs hand opening': 'hand imagined movement hand attempted movement imagined hand opening',
        'Participant performs palmar grasp': 'hand imagined movement hand attempted movement imagined palmar grasp',
        'Participant performs lateral grasp': 'hand imagined movement hand attempted movement imagined lateral grasp',
    },
    "ErrP_BCI_NER": {
        # ErrP_BCI_NER
        'There is a feedback on the screen which indicates that the selected item is similar to the expected item': 'no error-related potential',
        'There is a feedback on the screen which indicates that the selected item is different from the expected item': 'error-related potential',
    },
    "ErrP_MERP": {
        # ErrP_MERP
        'Target is located in the left': 'target left',
        'Target is located in the right': 'target right',
        'The cursor moves to the left': 'cursor left',
        'The cursor moves to the right': 'cursor right',
        'The cursor moves to correct diretion': 'no error-related potential',
        'The cursor moves to wrong direction': 'error-related potential',
    },
    "RS_ALPHA": {
        "Participant keeps closing eyes": "eye no movement eye closed",
        "Participant keeps opening eyes": "eye no movement eye open",
    },
    "RS_SPIS": {
        "Participant is in resting state and keeps closing eyes": "eye no movement eye closed",
        "Participant is in resting state and trys to keep opening eyes": "eye no movement eye open",
    }
}

# datasets = ["MI_Limb", "MI_BBCI_IV_Berlin", "MI_BBCI_IV_Graz_a", "MI_BBCI_IV_Graz_b", "MI_II"]
datasets = ["MI_LR", "MI_eegmmidb", "MI_HGD", "MI_Two", "MI_SCP", "MI_GVH", "MI_GAL", "MI_ULM", "MI_SCI", "RS_ALPHA", "RS_SPIS"]
#datasets = ["RS_SPIS"]
prepared_data_directory = "/itet-stor/kard/deepeye_storage/foundation_prepared/"

mi_overall = {"train": [], "test": []}
mi_overall = json.load(open("/itet-stor/kard/net_scratch/eeg-foundation/src/preparation/mi_all.json"))
print("Starting creating task...")

for dataset in datasets:
    memory = set([])
    print(dataset)
    all_units = sorted(os.listdir(prepared_data_directory))
    load_file_pattern = re.compile(f"(?=.*{dataset})")
    all_units = [unit for unit in all_units if load_file_pattern.match(unit)]
    labels = defaultdict(lambda: None)
    for i, unit in tqdm(enumerate(all_units), total=len(all_units)):
        data = pd.read_pickle(prepared_data_directory + unit)
        non_empty_indices = data[(data['Event Description'] != '') & (data.index.to_series().apply(lambda x: isinstance(x, int)))].index.tolist()
        # Append the length of the DataFrame to handle the last range
        non_empty_indices.append(len(data) - 1) # -1 because some strange cases
        descriptions = []
        for j, index in enumerate(non_empty_indices[:-1]):
            label = data.at[index, 'Event Description']
            if isinstance(label, str):
                label = label.split(":")[-1]
            else:
                label = ''
            label = label.strip()
            if label not in memory:
                memory.add(label)
                # print(label)
            descriptions.append((label, index, non_empty_indices[j + 1]))
        # print(unit, descriptions)
        a = set(data.loc[:, 'Source file'].tolist())
        source_file = [value for value in a if value][0]
        for (description, start, end) in descriptions:
            start_seconds = float(data["time in seconds"].loc[start])
            if dataset == "MI_LR":
                end_seconds = float(data["time in seconds"].loc[end - 1])
            else:
                end_seconds = float(data["time in seconds"].loc[end])
            length_seconds = end_seconds - start_seconds
            if description in map_labels[dataset]:
                if dataset in split["train"] and [el in source_file for el in split["train"][dataset]].count(True) > 0:
                    mi_overall["train"].append({"input": [unit], "label": map_labels[dataset][description], "length_seconds": float(length_seconds), "start": start, "end": end})
                if dataset in split["test"] and [el in source_file for el in split["test"][dataset]].count(True) > 0:
                    mi_overall["test"].append({"input": [unit], "label": map_labels[dataset][description], "length_seconds": float(length_seconds), "start": start, "end": end})
    # Save as a json the body part task
    with open("mi_all.json", "w") as f:
       json.dump(mi_overall, f)

