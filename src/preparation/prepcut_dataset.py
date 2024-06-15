import sys
from prepcutter import PrepCutter
from datasets.mi_limb import mi_limb_read_file
from datasets.mi_lr import mi_lr_read_file
from datasets.mi_bci_iv_berlin import mi_bci_iv_berlin_read_file
from datasets.mi_bci_iv_graz_a import mi_bci_iv_graz_a_read_file
from datasets.mi_bci_iv_graz_b import mi_bci_iv_graz_b_read_file
from datasets.mi_eegmmidb import mi_eegmmidb_read_file
from datasets.mi_hgd import mi_hgd_read_file
from datasets.mi_two import mi_two_read_file
from datasets.mi_ii import mi_ii_read_file
from datasets.mi_scp import mi_scp_read_file
from datasets.mi_gvh import mi_gvh_read_file



def prepcut_MI_Limb(verbose=False):
    prepcutter = PrepCutter(dataset="MI_Limb",
                        load_directory="/itet-stor/kard/deepeye_storage/foundation/MI_Limb/",
                        save_directory="/itet-stor/kard/deepeye_storage/foundationALL_prepcut/",
                        load_file_pattern=r'.*\.mat$',
                        read_file=mi_limb_read_file,
                        override=False,
                        verbose=verbose, 
                        preprocess=False)
    prepcutter.process_dataset()

def prepcut_MI_LR(verbose=False):
    prepcutter = PrepCutter(dataset="MI_LR",
                        load_directory="/itet-stor/kard/deepeye_storage/foundation/MI_LR/",
                        save_directory="/itet-stor/kard/deepeye_storage/foundationALL_prepcut/",
                        load_file_pattern=r'.*\.mat$',
                        read_file=mi_lr_read_file,
                        override=False,
                        verbose=False, 
                        preprocess=False)
    prepcutter.process_dataset()


def prepcut_BCI_IV_Berlin(verbose=False):
    prepcutter = PrepCutter(dataset="MI_BCI_IV_Berlin",
                        load_directory="/itet-stor/kard/deepeye_storage/foundation/BBCI_IV/Berlin",
                        save_directory="/itet-stor/kard/deepeye_storage/foundationALL_prepcut/",
                        load_file_pattern=r'.*\.mat$',
                        read_file=mi_bci_iv_berlin_read_file,
                        override=False,
                        verbose=False, 
                        preprocess=False)
    prepcutter.process_dataset()

def prepcut_BCI_IV_Graz_a(verbose=False):
    prepcutter = PrepCutter(dataset="MI_BCI_IV_Graz_a",
                        load_directory="/itet-stor/kard/deepeye_storage/foundation/BBCI_IV/Graz/a",
                        save_directory="/itet-stor/kard/deepeye_storage/foundationALL_prepcut/",
                        load_file_pattern=r'.*\.gdf$',
                        read_file=mi_bci_iv_graz_a_read_file,
                        override=False,
                        verbose=False, 
                        preprocess=False)
    prepcutter.process_dataset()

def prepcut_BCI_IV_Graz_b(verbose=False):
    prepcutter = PrepCutter(dataset="MI_BCI_IV_Graz_b",
                        load_directory="/itet-stor/kard/deepeye_storage/foundation/BBCI_IV/Graz/b",
                        save_directory="/itet-stor/kard/deepeye_storage/foundationALL_prepcut/",
                        load_file_pattern=r'.*\.gdf$',
                        read_file=mi_bci_iv_graz_b_read_file,
                        override=False,
                        verbose=False, 
                        preprocess=False)
    prepcutter.process_dataset()

def prepcut_MI_eegmmidb(verbose=False):
    prepcutter = PrepCutter(dataset="MI_eegmmidb",
                        load_directory="/itet-stor/kard/deepeye_storage/foundation/eegmmidb/",
                        save_directory="/itet-stor/kard/deepeye_storage/foundationALL_prepcut/",
                        load_file_pattern=r'.*\.edf$',
                        read_file=mi_eegmmidb_read_file,
                        override=False,
                        verbose=False, 
                        preprocess=False)
    prepcutter.process_dataset()

def prepcut_MI_HGD(verbose=False):
    prepcutter = PrepCutter(dataset="MI_HGD",
                        load_directory="/itet-stor/kard/deepeye_storage/foundation/MI_HGD/",
                        save_directory="/itet-stor/kard/deepeye_storage/foundationALL_prepcut/",
                        load_file_pattern=r'.*\.edf$',
                        read_file=mi_hgd_read_file,
                        override=False,
                        verbose=False, 
                        preprocess=False)
    prepcutter.process_dataset()

def prepcut_MI_Two(verbose=False):
    prepcutter = PrepCutter(dataset="MI_Two",
                        load_directory="/itet-stor/kard/deepeye_storage/foundation/MI_Two/",
                        save_directory="/itet-stor/kard/deepeye_storage/foundationALL_prepcut/",
                        load_file_pattern=r'.*\.mat$',
                        read_file=mi_two_read_file,
                        override=False,
                        verbose=False, 
                        preprocess=False)
    prepcutter.process_dataset()

def prepcut_MI_II(verbose=False):
    prepcutter = PrepCutter(dataset="MI_II",
                        load_directory="/itet-stor/kard/deepeye_storage/foundation/MI_II/",
                        save_directory="/itet-stor/kard/deepeye_storage/foundationALL_prepcut/",
                        load_file_pattern=r'.*\.mat$',
                        read_file=mi_ii_read_file,
                        override=False,
                        verbose=False, 
                        preprocess=False)
    prepcutter.process_dataset()

def prepcut_MI_SCP(verbose=False):
    prepcutter = PrepCutter(dataset="MI_SCP",
                        load_directory="/itet-stor/kard/deepeye_storage/foundation/MI_SCP/",
                        save_directory="/itet-stor/kard/deepeye_storage/foundationALL_prepcut/",
                        load_file_pattern=r'.*\.mat$',
                        read_file=mi_scp_read_file,
                        override=False,
                        verbose=False, 
                        preprocess=False)
    prepcutter.process_dataset()

def prepcut_MI_GVH(verbose=False):
    prepcutter = PrepCutter(dataset="MI_GVH",
                        load_directory="/itet-stor/kard/deepeye_storage/foundation/MI_GVH/",
                        save_directory="/itet-stor/kard/deepeye_storage/foundationALL_prepcut/",
                        load_file_pattern=r'.*\.mat$',
                        read_file=mi_gvh_read_file,
                        override=False,
                        verbose=False, 
                        preprocess=False)
    prepcutter.process_dataset()


def prepcut_dataset(dataset_name):
    dataset_dict = {
        "MI_Limb": lambda: prepcut_MI_Limb(verbose=False),
        "MI_LR": lambda: prepcut_MI_LR(verbose=False),
        "MI_BCI_IV_Berlin": lambda: prepcut_BCI_IV_Berlin(verbose=False),
        "MI_BCI_IV_Graz_a": lambda: prepcut_BCI_IV_Graz_a(verbose=False),
        "MI_BCI_IV_Graz_b": lambda: prepcut_BCI_IV_Graz_b(verbose=False),
        "MI_eegmmidb": lambda: prepcut_MI_eegmmidb(verbose=False),
        "MI_HGD": lambda: prepcut_MI_HGD(verbose=False),
        "MI_Two": lambda: prepcut_MI_Two(verbose=False),
        "MI_II": lambda: prepcut_MI_II(verbose=False),
        "MI_SCP": lambda: prepcut_MI_SCP(verbose=False),
        "MI_GVH": lambda: prepcut_MI_GVH(verbose=False)
    }
    
    if dataset_name in dataset_dict:
        dataset_dict[dataset_name]()
    else:
        print("Dataset '{}' is not supported. Try integrating the dataset in the interface by following the guidelines in README.".format(dataset_name))


if __name__ == "__main__":
    dataset = sys.argv[1]
    prepcut_dataset(dataset)


