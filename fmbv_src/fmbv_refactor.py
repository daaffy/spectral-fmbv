import SimpleITK as sitk
import GETagAnalyser
import numpy as np
import copy
from sklearn.linear_model import LinearRegression
from scipy import ndimage

'''
    NOTE (TO-DO):
- load_kretz is used to get metadata from .vol files. How does the pipeline change for .dcm files?
- Only accepts seg vol_id = 1. Any other numbers (non-zero) get converted into vol_id = 1 (_clean_segmentation).
'''

class FMBV:
    ''' 
        FMBV
    Object requires power doppler and segmentation data to computer FMBV related quanties.
    '''

    def __init__(self, *args, **kwargs):
        '''
        
            mode
        0 - Calls original FMBV functions
        1 - Calls refactor of original FMBV functions
        2 - Calls modified version of FMBV functions
        '''

        # Default 
        self.mode = 0                   # Select processing mode. 
        self.verbose = False            # = True; print what's going on.
        self.pre_scale = 1
        self.scale = 1.00               # Scale final standardised image to this value.
        self.pd_low_threshold = 2.0     # Voxels values below this value are discarded in FMBV processing
        self.combine = True             # Ignore segmentation partitions and treat all non-zero indices as a whole segmentation
        self.vol_id = 1                 # 
        self.dc_thickness = 2
        self.max_pixel_value = 1.0
        self.dp_img, self.seg_img = None, None
        self.pd_supplied = False
        self.seg_supplied = False
        self.kretz_supplied = False
        self.vis_information = None
        self.clip = True
        self.std_regress = False
        self.original_seg_voxel_nums = 0 # this needs better treatment

        # Dealing with large files
        self.mm = False                 # Minimize memory? We run into trouble using FMBV (_refactor?) on large PD files etc.
        self.zoom = 1

        # Visualisation Dictionaries
        self.global_figdata_std_1, self.global_figdata_std_2 = None, None
        self.dc_figdata_std_1, self.dc_figdata_std_2 = [], []
        self.dc_std_values_1 = []
        self.dc_std_values_2 = []
        self.dc_depth_values = []
        self.vxl_nums = []

        self.percs = [0.99, 0.97, 0.95, 0.90, 0.85, 0.8]
        self.fmbvs_naive = {
            "mpi" : np.nan,
            "max" : np.nan,
            "percs" : np.empty(len(self.percs))
        }
        self.fmbvs_naive["percs"][:] = np.nan

        # Handle kwargs
        if "verbose" in kwargs:
            self.verbose = kwargs["verbose"]
        if "mode" in kwargs:
            self.mode = kwargs["mode"]
        if "scale" in kwargs:
            self.scale = kwargs["scale"]
        if "vol_id" in kwargs:
            # We assume here that the user intends on selecting a specific index to perform FMBV on
            self.combine = False
            self.vol_id = kwargs["vol_id"]
        if "max_pixel_value" in kwargs:
            self.max_pixel_value = kwargs["max_pixel_value"]
        if "pd_low_threshold" in kwargs:
            self.pd_low_threshold = kwargs["pd_low_threshold"]
        if "pre_scale" in kwargs:
            self.pre_scale = kwargs["pre_scale"]
        if "clip" in kwargs:
            self.clip = kwargs["clip"]
        if "std_regress" in kwargs:
            self.std_regress = kwargs["std_regress"]
        if "mm" in kwargs:
            self.mm = kwargs["mm"]
        if "zoom" in kwargs:
            self.zoom = kwargs["zoom"]

        self.vrbs("Initialising FMBV object...")
        self.vrbs("Mode: " + str(self.mode))

        if self.mm:
            self.vrbs("Minimise memory mode!")
        else:
            self.vrbs("Default memory handling.")

        # Load Power Doppler Data (necessary)

        # Load Segmentation Data
        # ... if not, treat whole space as segmentation

        # Load Kretz
        # ... if not, we cannot perform depth-corrected FMBV

    # --- PROCESS FMBV
    def global_method(self, *args, skip_standardisation=False):
        '''
            NOTE (TO-DO):
        - Add vrbs.
        '''
        if args: # non-empty args; do we want to do this??
            # if (args[0]) ... CHECKS HERE!!
            current_seg = args[0]
        else:
            if not self.seg_supplied:
                self._set_default_segmentation(mode="sweep")

            current_seg = self.seg_array

        self._check_pd_seg_sizes()
        self._clean_segmentation() # Might change later when we allow for regions

        pd_flat = self.pd_array.flatten().astype(float)
        seg_flat = current_seg.flatten().astype(int)
        # mask = (seg_flat > 0) # (!!!!) for SNG paper

        mask = (seg_flat == self.vol_id)

        pd_trim = pd_flat[mask]

        # self.std_value_1, self.std_value_2 = self.std_method(pd_trim[pd_trim > self.pd_low_threshold])
        if not skip_standardisation:
            self.global_std_value_1, self.global_std_value_2, self.global_figdata_std_1, self.global_figdata_std_2 = self.std_method(pd_trim[pd_trim > self.pd_low_threshold])
            self.global_fmbv_value_1 = self.normalise(copy.deepcopy(pd_trim), self.global_std_value_1)
            self.global_fmbv_value_2 = self.normalise(copy.deepcopy(pd_trim), self.global_std_value_2)
        else:
            self.global_std_value_1 = self.max_pixel_value
            self.global_std_value_2 =  self.global_std_value_1
            self.global_fmbv_value_1 = self.normalise(copy.deepcopy(pd_trim), self.max_pixel_value)
            self.global_fmbv_value_2 = self.global_fmbv_value_1

        if not self.mm: # e.g., if mm=True (we want to minimize memory) we ignore the storage of large arrays.
            self.pd_std_global_array = np.multiply(self.pd_array, current_seg/self.global_std_value_2)
            if self.clip:
                self.vrbs("Clipping at std value...")
                self.pd_std_global_array[self.pd_std_global_array >= 1] = 1.0

            self.pd_std_global_array = self.pd_std_global_array*self.max_pixel_value

    def depth_corrected_method(self):
        '''
            depth_corrected_method
        Apply standardisation in radial stratifications to account for attenuation vs. tissue depth.
        '''
        
        if not self.kretz_supplied:
            raise Exception("Kretz data has not been supplied. Use load_kretz.")

        self.vrbs("Applying depth correction.")

        self._check_pd_seg_sizes()
        self._clean_segmentation() # Might change later when we allow for regions

        distance = self.get_distance_map()
        self.vrbs("Creating img_distance array...")
        img_distance = self.seg_array * distance
        if not self.mm:
            self.layers = img_distance

        pd_flat = self.pd_array.flatten().astype(int)
        seg_flat = img_distance.flatten().astype(int)
        ma = seg_flat[seg_flat != 0]

        self.start_depth = np.min(ma)
        self.end_depth = np.max(ma)
        self.seg_volume = self.original_seg_voxel_nums * self.bmode_spacing**3 # mm^3

        c = self.dc_thickness

        self.vrbs("--- ENTERING THE ONION LOOP")
        fmbv_vals, vxl_nums = [], []
        if not self.mm:
            self.pd_std_array = np.zeros(np.shape(self.pd_array))
        
        success_num = 0
        layer_range = np.arange(ma.min(), ma.max(), c)
        self.numslices = len(layer_range)
        for i in layer_range:
            self.vrbs('\n Layer '+ str(i) + ' out of '+ str(ma.max()-1))

            mask = (seg_flat >= i) & (seg_flat <= i + c)
            array_mask = (img_distance >= i) & (img_distance <= i + c)
            # pd_fmbv = pd_flat[mask].tolist()
            pd_fmbv = pd_flat[mask]
            vxl_num = sum(mask)

            std_value_1, std_value_2, temp1, temp2 = self.std_method(pd_fmbv[pd_fmbv > self.pd_low_threshold])
            self.vrbs('std value for current layer: ' + str(std_value_2))

            self.dc_figdata_std_1.append(temp1)
            self.dc_figdata_std_2.append(temp2)
            self.dc_std_values_1.append(std_value_1)
            self.dc_std_values_2.append(std_value_2)
            self.dc_depth_values.append(i)

            try: 
                fmbv_value = np.nan
                if std_value_2 is not None:
                    fmbv_value = self.normalise(copy.deepcopy(pd_fmbv),std_value_2)
                    if not self.mm:
                        self.pd_std_array = self.pd_std_array + np.multiply(self.pd_array, array_mask/std_value_2)
                success_num = success_num + 1
            except:
                fmbv_value = np.nan
            fmbv_vals.append(fmbv_value)
            vxl_nums.append(vxl_num)

        self.vrbs("--- EXITING THE ONION LOOP")
        self.success_num = int(success_num / self.numslices * 100)

        # if self.std_regress:
        #     self.vrbs("Using regressed std values for standardisation!")

        # self.std_value_2_reg = self.do_std_regress(self.dc_depth_values,self.dc_std_values_2)
        # for i in range(len(self.dc_std_values_2)):
        #     # which std value to use??
        #     if self.std_regress:
        #         std_value_2 = self.std_value_2_reg[i]
        #     else:
        #         std_value_2 = self.dc_std_values_2[i]

        #     array_mask = (img_distance >= i) & (img_distance <= i + c)
        #     try: 
        #         fmbv_value = np.nan
        #         if std_value_2 is not None:
        #             fmbv_value = self.normalise(copy.deepcopy(pd_fmbv),std_value_2)
        #             self.pd_std_array = self.pd_std_array + np.multiply(self.pd_array, array_mask/std_value_2)
        #     except:
        #         fmbv_value = np.nan
        #     fmbv_vals.append(fmbv_value)


        self.dc_std_values_2_pre_regress = self.dc_std_values_2
        
        if self.mode == 1:
            if self.mm:
                raise Exception("Mode 1 banned (mm=True).")
            # mode = 1:
            # don't average the fmbv_vals!!!
            # we perform a weighted average!!!
            self.vrbs("FMBV values per layer: ")

            # recalculate fmbv using regressed values
            # note: no regression on first std, could perhaps improve here...

            std_value_regress = self.do_std_regress(self.dc_depth_values,self.dc_std_values_2, weights = vxl_nums)

            fmbv_vals = []
            self.dc_std_values_2 = []
            self.pd_std_array = np.zeros(np.shape(self.pd_array))
            for i in range(len(self.dc_depth_values)):

                d = self.dc_depth_values[i]
                mask = (seg_flat >= d) & (seg_flat <= d + c)
                array_mask = (img_distance >= d) & (img_distance <= d + c)
                pd_fmbv = pd_flat[mask]

                try: 
                    fmbv_value = np.nan
                    if std_value_2 is not None:
                        fmbv_value = self.normalise(copy.deepcopy(pd_fmbv),std_value_regress[i])
                        self.pd_std_array = self.pd_std_array + np.multiply(self.pd_array, array_mask/std_value_regress[i])
                except:
                    fmbv_value = np.nan

                self.dc_std_values_2.append(std_value_regress[i])
                fmbv_vals.append(fmbv_value)


            # self.vrbs(fmbv_vals)
            self.vrbs("Vxl number per layer: ")
            self.vrbs(vxl_nums)
            self.dc_fmbv = np.sum(np.array(fmbv_vals)*np.array(vxl_nums))/np.sum(vxl_nums)
            self.vrbs("FMBV-DC: " + str(self.dc_fmbv))
            self.fmbv_values = len(fmbv_vals)*np.array(fmbv_vals)*np.array(vxl_nums)/np.sum(vxl_nums)
            self.vxl_nums = vxl_nums
        else:
            # mode = 0:
            self.vrbs("FMBV values per layer: ")
            self.vrbs(fmbv_vals)
            self.dc_fmbv = np.nanmean(fmbv_vals)
            self.vrbs("FMBV-DC: " + str(self.dc_fmbv))
            self.vxl_nums = vxl_nums
            self.fmbv_values = fmbv_vals


        # Clip values
        if not self.mm:
            if self.clip:
                self.vrbs("Clipping at std value...")
                self.pd_std_array[self.pd_std_array >= 1] = 1.0

            self.pd_std_array = self.pd_std_array*self.max_pixel_value

            self.vrbs("Maximum pixel value is: " + str(np.max(self.pd_std_array)))

            # Create image
            if self.zoom == 1: # otherwise cannot copy metadata... should be a way around this.
                self.pd_std_img = sitk.GetImageFromArray(self.pd_std_array)
                self.pd_std_img.CopyInformation(self.pd_img)
                self.pd_std_img.SetMetaData('standardised','True') # HasMetaDataKey

    def naive_fmbvs(self, *args):
        '''
            naive_fmbvs
        Here we use some very basic standardisations as a baseline to compare other stnadardisation methods.
        '''

        if args: # non-empty args; do we want to do this??
            # if (args[0]) ... CHECKS HERE!!
            current_seg = args[0]
        else:
            if not self.seg_supplied:
                self._set_default_segmentation(mode="sweep")

            current_seg = self.seg_array

        self._check_pd_seg_sizes()
        self._clean_segmentation() # Might change later when we allow for regions

        pd_flat = self.pd_array.flatten().astype(float)
        seg_flat = current_seg.flatten().astype(int)
        # mask = (seg_flat > 0) # (!!!!) for SNG paper

        mask = (seg_flat == self.vol_id)

        pd_trim = pd_flat[mask].astype(int)

        pd_trim = pd_trim[pd_trim > self.pd_low_threshold]

        # print(pd_trim)

        mpi = np.mean(pd_trim)
        maxstd = self.normalise(copy.deepcopy(pd_trim), np.max(pd_trim))
        
        # hist, x_axis = np.histogram(vals,bins=np.max(vals)+1-np.min(vals))
        hist, _ = np.histogram(pd_trim,bins=range(0,np.max(pd_trim)+1))
        cumsum = np.cumsum(hist)
        cumsum = cumsum / np.max(cumsum)

        self.vrbs("pd max: "+str(np.max(pd_trim)))
        
        self.perc_stds = []
        self.perc_fmbvs = []

        x_axis = range(0,np.max(pd_trim))
        self.temp_cumsum = cumsum
        # cumulative distribution structure
        for p in self.percs:
            i = np.argwhere(cumsum > p)[0][0]
            std_p = x_axis[i]

            self.perc_stds.append(std_p)
            self.perc_fmbvs.append(self.normalise(copy.deepcopy(pd_trim), std_p))

        # self.fmbvs_naive = []
        self.fmbvs_naive = {
            "mpi" : mpi,
            "max" : maxstd,
            "percs" : self.perc_fmbvs
        }


        # self.fmbvs_naive["mpi"] = np.mean(pd_trim)
        # self.fmbvs_naive["max"] = np.normalise(copy.deepcopy(pd_trim), np.max(pd_trim))

        0

    # --- STANDARDISATION FUNCTIONS

    def std_method(self, pd_data):
        ''' 
            std_method
        Redirect to the appropriate std_method_#
        '''

        self.vrbs("Calculating standardisation.")

        match self.mode:
            case 0:
                return self._std_method_0(pd_data, return_visualisation_data = True)
            case 1:
                return self._std_method_1(pd_data, return_visualisation_data = True)
            case _:
                raise Exception("Mode " + str(self.mode) + " not recognised.")

    def _std_method_0(self, pd_data, return_visualisation_data = False):
        '''
                std_method_0
            Refactor of original standardisation method for modularity and visualisation.
            Actual algorithm is largely unchanged.

            pd_data - Must be "flat."
        '''

        pd_data = pd_data.astype(int)

        if not return_visualisation_data:
            # In case someone wants to use _std_method_0 as a simple utility and ignore visualisation data.
            std_value1, std_value2, _, _ = self._std_method_0(pd_data, return_visualisation_data=True)
            return std_value1, std_value2

        self.vrbs("I'm in std_method_0.")

        fig_data_std_1 = fig_data_std_2 = {
                "complete": False,

                "mode": self.mode,
                "std_value": None
            }
        
        refImageScalars = np.array(pd_data)

        # Stage 1:
        try:
            refImageScalars = refImageScalars[refImageScalars > self.pd_low_threshold]
            scalarWMF = int(self.pd_low_threshold)
            scalarmin = np.int64(0)
            scalarmax = np.int64(np.ceil((np.max(refImageScalars))))
            scalar_range = (scalarmax - scalarmin) +1        
            xaxis = np.zeros(scalar_range) 
            yaxis = np.zeros(scalar_range)
            std_value1, std_value2 = None, None
            step2_calculated = False

            for a in range(1,scalar_range+1):            
                xaxis[a-1] = a      
            #print("Calculating FMBV....")
                        
            for a in range(1,np.size(refImageScalars)):
                yaxis[refImageScalars[a]] = yaxis[refImageScalars[a]] + 1
            
            #produce cumulative summing of these values
            #plot the cumulative pdf of the ROI.
            cumsum = np.cumsum(yaxis)
            stg1_cumsum = cumsum


            fig_data_std_1["x"] = xaxis
            fig_data_std_1["y_hist"] = yaxis
            fig_data_std_1["y_cumsum"] = cumsum


            # print(xaxis,cumsum) # !!
            (ar,br) = np.polyfit(xaxis, cumsum, 1)
            linfit = np.polyval([ar,br],xaxis)
            stg1_linfit = linfit

            fig_data_std_1["y_linfit_0"] = linfit, # line of best fit to calculate intersection points
            fig_data_std_1["y_linfit_0"] = fig_data_std_1["y_linfit_0"][0] # ...[0] to look inside the tuple structure. I'm not sure why we have to do it like this but its the only way I could get it to work???

            #start from where gradient begins
            st = scalarmin               
            #absscia turning pt calculation
            absscia = cumsum - linfit
            stg1_absscia = absscia

            st_i = 0
            while cumsum[st_i] < 1:
                st_i+=1
            
            #find tangent points
            absscianum = np.argmax(absscia[scalarWMF:])+scalarWMF 
            pt2 = np.nanmax([st_i,absscianum])
            
            if cumsum[pt2] < linfit[pt2]:
                while cumsum[pt2] < linfit[pt2]:
                    pt2 = pt2+1
            else:
                while cumsum[pt2] > linfit[pt2]:
                    pt2 = pt2+1

            #print st
            pt1 = np.nanmax([st_i,scalarWMF])
            while cumsum[pt1] < linfit[pt1]:               
                pt1 = pt1+1

            stg1_pt2 = pt2
            stg1_pt1 = pt1

            fig_data_std_1["pt_a"] =  xaxis[pt1]
            fig_data_std_1["pt_b"] = xaxis[pt2]

            #1st tangent linear regression on pts +/- 10 away..
            #bounds checking
            if pt1 > scalarmin+11:                
                coeffspt1 = np.polyfit(xaxis[pt1-10:pt1+10], cumsum[pt1-10:pt1+10],1)
            else:
                diff = 10-pt1
                coeffspt1 = np.polyfit(xaxis[pt1-10+diff:pt1+10], cumsum[pt1-10+diff:pt1+10],1)
                
            #run the tangent from the point of intersection up to the 2nd intersection point
            pt1linfit = np.polyval(coeffspt1,xaxis[pt1:pt2] )
            stg1_pt1linfit = pt1linfit

            fig_data_std_1["y_linfit_a"] = np.polyval(coeffspt1,xaxis)

            if pt2 < scalarmax-11 :                
                coeffspt2 = np.polyfit(xaxis[pt2-10:pt2+10], cumsum[pt2-10:pt2+10],1)
                pt2linfit = np.polyval(coeffspt2,xaxis[pt1:pt2+10] )

            else:
                diff = 10-pt2
                coeffspt2 = np.polyfit(xaxis[pt2-10:pt2+10-diff], cumsum[pt2-10:pt2+10-diff],1)    
                pt2linfit = np.polyval(coeffspt2,xaxis[pt1:pt2+10-diff] )
            stg1_pt2linfit = pt2linfit

            fig_data_std_1["y_linfit_b"] = np.polyval(coeffspt2,xaxis)

            i=0
            while pt1linfit[i] < pt2linfit[i]:
                i = i+1
            i = pt1 + i

            pos_ndg = 10+scalarWMF
            neg_ndg = 10-scalarWMF
            
            coeffsabs = np.polyfit( xaxis[np.argmax(absscia[scalarWMF:])-neg_ndg:np.argmax(absscia[scalarWMF:])+pos_ndg], absscia[np.argmax(absscia[scalarWMF:])-neg_ndg:np.argmax(absscia[scalarWMF:])+pos_ndg] ,2)
            polyfitabs = np.polyval(coeffsabs,absscia[np.argmax(absscia[scalarWMF:]-neg_ndg):np.argmax(absscia[scalarWMF:])+pos_ndg])  

            std_value1 = min([absscianum,i])  #+ scalarWMF

            self.vrbs("std_value_1 = "+ str(std_value1))

            fig_data_std_1["std_value"] = std_value1
            fig_data_std_1["complete"] = True

            # Figure Data
            # fig_data_std_1 = {
            #     "complete": True,

            #     "x": xaxis,
            #     "y_hist": yaxis,
            #     "y_cumsum": cumsum,

            #     "y_linfit_0": linfit, # line of best fit to calculate intersection points
            #     "pt_a": xaxis[pt1],
            #     "pt_b": xaxis[pt2],

            #     "y_linfit_a": np.polyval(coeffspt1,xaxis),
            #     "y_linfit_b": np.polyval(coeffspt2,xaxis),

            #     "std_value": std_value1
            # }
            
        except Exception as e:
            print(e)
            std_value1 = None


        # Stage 2:
        try:
            #convert vector of n-scalars into binned sv->256 length vector
            sv = std_value1        
            yaxis = np.zeros(scalar_range-sv)
            for a in range(1,np.size(refImageScalars)):
                if refImageScalars[a] > sv:
                    yaxis[refImageScalars[a] - sv] = yaxis[refImageScalars[a]- sv] +1

            xaxissv = np.zeros(scalar_range-sv) 
            for a in range(1,(scalar_range+1)-sv):            
                xaxissv[a-1] = a 
            stg2_xaxis = xaxissv
            #produce cumulative summing of these values
            cumsum = np.cumsum(yaxis)
            stg2_cumsum = cumsum

            (ar,br) = np.polyfit(xaxissv, cumsum, 1)
            linfit = np.polyval([ar,br],xaxissv)
            stg2_linfit = linfit

            #absscia turning pt calculation
            absscia = cumsum - linfit
            stg2_absscia = absscia
            abssicianum = np.argmax(absscia)

            #find tangent points
            pt2 = abssicianum
            while cumsum[pt2] > linfit[pt2]:                        
                if pt2 >= min(len(cumsum)-5, len(linfit)-5):
                    break
                pt2 = pt2+1
                #print(pt2, abssicianum, len(linfit), len(cumsum))
            stg2_pt2 = pt2

            pt1 = 0
            while cumsum[pt1] < linfit[pt1]:            
                pt1 = pt1+1
            stg2_pt1 = pt1

            #out of bounds checks -- lower bounds
            if pt1 > scalarmin+10:                
                coeffspt1 = np.polyfit(xaxis[pt1-10:pt1+10], cumsum[pt1-10:pt1+10],1)
            else:
                diff = 10-pt1
                coeffspt1 = np.polyfit(xaxis[pt1-10+diff:pt1+10], cumsum[pt1-10+diff:pt1+10],1)

            #1st tangent linear regression on pts +/- 10 away..
            #coeffspt1 = polyfit(xaxissv[pt1-10:pt1+10], cumsum[pt1-10:pt1+10],1)
            #run the tangent from the point of intersection up to the 2nd intersection point
            pt1linfit = np.polyval(coeffspt1,xaxissv[pt1:pt2] )
            stg2_pt1linfit = pt1linfit

            if pt2 < scalarmax-11 :                
                coeffspt2 = np.polyfit(xaxissv[pt2-10:pt2+10], cumsum[pt2-10:pt2+10],1)
                pt2linfit = np.polyval(coeffspt2,xaxissv[pt1:pt2+10] )
            else:
                diff = 10-pt2
                coeffspt2 = np.polyfit(xaxissv[pt2-10:pt2+10-diff], cumsum[pt2-10:pt2+10-diff],1)
                pt2linfit = np.polyval(coeffspt2,xaxissv[pt1:pt2+10-diff] )
            stg2_pt2linfit = pt2linfit

            #need to fit 2nd order polynomial to this point +/-
            i=0 
            while pt1linfit[i] < pt2linfit[i]:
                i = i+1
            i = pt1 + i            

            #coeffsabs = np.polyfit( xaxissv[np.argmax(absscia)-10:np.argmax(absscia)+10], absscia[np.argmax(absscia)-10:np.argmax(absscia)+10] ,2)
            #polyfitabs = np.polyval(coeffsabs,absscia[np.argmax(absscia)-10:np.argmax(absscia)+10])
            std_value2 =  np.nanmin([abssicianum,i]) + std_value1
            step2_calculated = True

            self.vrbs("std_value_2 = " + str(std_value2))

            fig_data_std_2 = {
                "complete": True,

                "x": xaxissv,
                "y_hist": yaxis,
                "y_cumsum": cumsum,

                "y_linfit_0": linfit, # line of best fit to calculate intersection points
                "pt_a": xaxissv[pt1],
                "pt_b": xaxissv[pt2],

                "y_linfit_a": np.polyval(coeffspt1,xaxissv),
                "y_linfit_b": np.polyval(coeffspt2,xaxissv),

                "std_value": np.nanmin([abssicianum,i])
            }
            
        except (ValueError, IndexError, TypeError, UnboundLocalError) as e:
            # print('\r{}'.format(e), end = '')
            # print ("\rFMBV Failed at Step 1!", end='')
            # print(e)
            std_value2 = None

            # fig_data_std_2 = {
            #     "complete": False,

            #     "std_value": None
            # }
            # pass

        return std_value1, std_value2, fig_data_std_1, fig_data_std_2
        
    def _std_method_1(self, pd_data, return_visualisation_data = False):
        ''' 
            std_method_1
        Modified FMBV standardisation function.

        Here we modularise the process of finding tangents to the cumulative distribution.
        Next we use linear regression to smooth the standardisation values to minimise the
        effect of "vascular pulling" of the cumulative distribution in images with large
        vascular heterogeneity. 
        In depth_corrected_method we perform a weighted average of layered FMBV values.

        pd_data - Must be "flat."
        '''

        pd_data = pd_data.astype(int)

        if not return_visualisation_data:
            # In case someone wants to use _std_method_0 as a simple utility and ignore visualisation data.
            std_value1, std_value2, _, _ = self._std_method_1(pd_data, return_visualisation_data=True)
            return std_value1, std_value2
        
        self.vrbs("I'm in std_method_1.")

        # fig_data_std_1 = fig_data_std_2 = {
        #         "complete": False,

        #         "mode": self.mode,
        #         "std_value": None
        #     }
        
        std_val1, std_val2 = None, None
        std_val1, fig_data_std_1  = self._extract_knee(pd_data)
        if not std_val1 == None:
            std_val2, fig_data_std_2  = self._extract_knee(pd_data[pd_data >= std_val1])
        else:
            fig_data_std_2 = fig_data_std_1

        if not std_val2 == None:
            std_val2 = std_val2
            # std_val2 = std_val2 + std_val1
        else:
            std_val2 = np.nan

        return std_val1, std_val2, fig_data_std_1, fig_data_std_2
        
    def _extract_knee(self, vals, w = 5):
        '''
        "Two-Tangent" algorithm for determining the "knee" of cumulative distribution function.
        vals : List (?) of input values e.g., power doppler voxel values.
        
        See Rubin et al., "Normalizing Fractional Moving Blood Volume Estimates with
        Power Doppler US: Defining a Stable Intravascular Point with the Cumulative Power
        Distribution Function."
        '''
        
        self.vrbs("I'm in _extract_knee.")

        fig_data = {
                "complete": False,

                "mode": self.mode,
                "std_value": None
            }

        try:
            x_axis = range(0,np.max(vals))
            # hist, x_axis = np.histogram(vals,bins=np.max(vals)+1-np.min(vals))
            hist, _ = np.histogram(vals,bins=range(0,np.max(vals)+1))
            cumsum = np.cumsum(hist)
            y_axis = cumsum

            fig_data["x"] = x_axis
            fig_data["y_hist"] = hist
            fig_data["y_cumsum"] = y_axis

            # print(len(x_axis))
            # print(len(hist))

            linfit = np.polyval(np.polyfit(
                [x_axis[i] for i in range(len(x_axis)) if hist[i] > 0],
                [y_axis[i] for i in range(len(y_axis)) if hist[i] > 0], 1), x_axis)

            # linfit = np.polyval(np.polyfit(x_axis, y_axis, 1), x_axis)

            fig_data["y_linfit_0"] = linfit

            # print(np.nonzero(np.array(vals)))
            min_val = np.min(vals[np.nonzero(np.array(vals))])
            # print(min_val)
            
            # min_val = 0

            tmp = (y_axis-linfit)
            block_inds = self.get_blocks((tmp >= 0) * (np.array(range(len(tmp))) >= min_val))
            # block_inds = self.get_blocks((tmp >= 0) * (hist > 0))
            self.vrbs("Blocks: "+str(len(block_inds)))

            # print((tmp >= 0) * (hist > 0))
            # print(block_inds)

            # above_xs = np.nonzero((tmp >= 0) * x_axis * (hist > 0))
            above_xs = block_inds[0]

            pt1 = np.min(above_xs)
            pt2 = np.max(above_xs)

            fig_data["pt_a"] = pt1
            fig_data["pt_b"] = pt2

            # perform bounds checks
            # ...

            inds1 = range(np.max([pt1-w-1,0]),np.min([pt1+w,len(x_axis)-1]))
            # np.array(range(pt1-w-1,pt1+w), dtype=int)
            pt1_linfit = np.polyval(np.polyfit(
                [x_axis[i] for i in inds1],
                [y_axis[i] for i in inds1],
                1),x_axis)
            
            fig_data["y_linfit_a"] = pt1_linfit

            inds2 = range(np.max([pt2-w-1,0]),np.min([pt2+w,len(x_axis)-1]))
            pt2_linfit = np.polyval(np.polyfit(
                [x_axis[i] for i in inds2],
                [y_axis[i] for i in inds2],
                1),x_axis)
            
            fig_data["y_linfit_b"] = pt2_linfit
            
            tmp2 = (pt1_linfit-pt2_linfit)
            pt3 = np.min(np.nonzero((tmp2 >= 0) * x_axis))
            
            fig_data = {
                "complete": False,

                "mode": self.mode,
                "std_value": None,

                "x": x_axis,
                "y_hist": hist,
                "y_cumsum": y_axis,

                "y_linfit_0": linfit,
                "pt_a": pt1,
                "pt_b": pt2,

                "y_linfit_a": pt1_linfit,
                "y_linfit_b": pt2_linfit,

                "std_value": pt3,

                # extra:
                "tmp": tmp
            }
        except Exception as e:
            print(e)
            pt3 = None

        return pt3, fig_data
    
    def get_blocks(self, bool_vec, land=True):
        '''
            get_blocks
        The difference between the line of best fit and the cumulative distribution
        often yields multiple "islands" or "blocks" that will confuse the algorithm.

        Here we isolate the islands...

        Islands are indicated by the presence of "land" (=True by default).
        '''

        blocks = []
        curr_block = []

        for i in range(len(bool_vec)):
            if bool_vec[i] == land:
                # now we are on land...
                curr_block.append(i)
            
            if not bool_vec[i] == land:
                # now we are not on land
                if not len(curr_block) == 0: # if we just left land...
                    blocks.append(curr_block)

                curr_block = []

        return blocks

    # def _intersection_walker(self,vals,start_ind=0,pos=True):
    #     # * This basic walker method assumes a level of regularity of the values (too much noise could identify an unexpected intersection point).
    #     # * The true intersection point lies somewhere +-1 of the output.

    #     if not isinstance(vals,list):
    #         raise TypeError

    #     # define walker direction 
    #     if (pos):
    #         step = +1
    #     else:
    #         step = -1

    #     start_polarity = np.sign(vals[start_ind])
    #     i = start_ind
    #     while (i >= 0) & (i < len(vals)) & (np.sign(vals[i])*start_polarity >= 0):
    #         i += step

    #     return i

    def _std_method_2(self):
        
        0

    # --- PRE-PROCESSING
    def _check_pd_seg_sizes(self):
        '''
            _check_pd_seg_sizes
        Ensures that power Doppler array and segmentation array are identical in size.
        This must be checked before methods are applied since the segmentation array acts as a mask over the power Doppler array.
        '''
        if not self.seg_supplied or not self.pd_supplied:
            raise Exception("Either segmentation or power Doppler data is missing.")
        
        if (not np.array_equiv(self.pd_array.shape, self.seg_array.shape)):
            raise Exception("Power doppler array size", self.pd_array.shape, " and segmentation array ", self.seg_array.shape, " do not have the same dimensions.")

    def _clean_segmentation(self):
        '''
            _clean_segmentation
        Ensures that segmentation data is specified by vol_id only. Any non-zero values will be altered to become vol_id.
        '''
        if self.seg_supplied:
            self.seg_array = self.vol_id * (self.seg_array != 0)
        else:
            raise Exception("Segmentation not supplied in the first place.")
        
    def _set_default_segmentation(self, mode="sweep", floor=0):
        '''
            _set_default_segmentation
        Set segmentation to a sweep defined by the power Doppler array (non-zero values *usually* define the curvilinear sweep of the ultrasound.)
        '''
        self.seg_array = self._get_default_segmentation(mode=mode, floor=floor)

        # if not self.zoom == 1: # Only if we are resampling...
        #     self.seg_array = np.round(self.seg_array).astype(int)

        self.seg_supplied = True

    def _get_default_segmentation(self, mode="sweep", floor=0):
        '''
            _get_default_segmentation
        Return a default segmentation
        mode:
            = "sweep"; a sweep defined by the power Doppler array (non-zero values *usually* define the curvilinear sweep of the ultrasound.)
            = "blank"; segmentation defined as the whole domain.
        '''
        if mode == "sweep":
            if (self.pd_supplied):
                return np.multiply(np.ones(self.pd_array.shape),(self.pd_array>floor))
            else:
                raise Exception("Power Doppler not supplied to base default segmentation from.")
        elif mode == "blank":
                return np.ones(self.pd_array.shape)
        
        
        
    def calculate_seg_volume(self):
        0

    # --- LOAD FUNCTIONS
    def _load_img(self, input):
        '''
            NOTE (TO-DO):
        - More complex file handling here e.g., .nii.gz, .dcm, ...
        '''
        if not isinstance(input, str):
            raise TypeError("Input path must be a string.")

        img = sitk.ReadImage(input)
        array = sitk.GetArrayFromImage(img)

        # Handle large images (resample technique)
        if not self.zoom == 1:
            array = ndimage.zoom(array, (self.zoom, self.zoom, self.zoom))

        return array, img

    def load_pd(self, arg1):

        self.vrbs("Loading Power Doppler...")
        
        if isinstance(arg1, str):
            if arg1 == '':
                return
        
            self.pd_array, self.pd_img = self._load_img(arg1)
        elif isinstance(arg1, np.ndarray):
            self.pd_array = arg1
            self.pd_img = None
        else:
            raise Exception("Invalid power doppler input.")
        
        self.pd_array = self.pd_array * self.pre_scale
        
        if not self.pd_img == None:
            if (self.pd_img.HasMetaDataKey("standardised")):
                self.vrbs('Imported image has been pre-standardised.')
        
        self.pd_supplied = True

    def load_seg(self, arg1):

        self.vrbs("Loading Segmentation...")


        if arg1 == '':
            return
        
        if isinstance(arg1, str):
            self.seg_array, self.seg_img = self._load_img(arg1)
        else:
            raise Exception("Invalid segmentation input.")
        
        if not self.zoom == 1: # Only if we are resampling...
            self.seg_array = np.round(self.seg_array).astype(int)

        self.original_seg_voxel_nums = np.sum(self.seg_array == self.vol_id)

        self.seg_supplied = True

    def load_kretz(self, arg1):

        self.vrbs("Loading Kretz...")

        if arg1 == '':
            return

        # Only metadata required from .vol file
        if isinstance(arg1, str):
            self.kretz_metadata = GETagAnalyser.GETagAnalyser(arg1)

            self.bmode_spacing = np.frombuffer(self.kretz_metadata.getDataFromTags(
                int('0xc100', 16), int('0x201', 16)), np.float64)[0] * 1000.0
        else:
            raise Exception("Invalid Kretz input.")
        self.kretz_supplied = True

     
     # --- EXPORT FUNCTIONS

    # --- EXPORT FUNCTIONS
    def export_standardised(self, output_path):
        self.vrbs(' Exporting standardised image as .nii.gz.')
        sitk.WriteImage(self.pd_std_img, output_path)
        self.vrbs('Export successful to output path '+str(output_path))

    # --- MISC. METHODS
    def normalise(self, vals, sv):
        vals = vals/float(sv)

        for i in range(np.size(vals)): # clip vals at 1.0
            if vals[i] >= 1.0:
                vals[i] = 1.0

        return np.nanmean(vals)*100 # this is the final fmbv index (as a %)

    def get_angle(self,x, y, z, sweep_radius):
        """
        given a 3D-coordinate return the radian angle of to origin
        pBAngle : in b-mode plane
        pVAngle : in b-mode plane

        x,y,z : physical position of the image co-ordinate
        sweep_radius : resolution * offset

        """
        if not self.mm:

            rDsq = sweep_radius ** 2

            xsq = x ** 2
            ysq = y ** 2
            zsq = z ** 2

            tworDz = 2 * sweep_radius * z
            tmpone = rDsq + ysq + zsq + tworDz
            tmptwo = xsq + ysq + zsq + tworDz + 2 * rDsq
            tmpthree = np.sqrt(tmpone) * 2 * sweep_radius

            pRB = np.sqrt(tmptwo - tmpthree)
            pBAngle = np.arcsin(x / pRB)
            pVAngle = np.arcsin(y / (sweep_radius + pRB * np.cos(pBAngle)))
        
        else: # minimise memory; this method gets hung up on large files!
            sh = np.shape(x)
            pRB = np.zeros(sh)

            # print(sh)
            # print(np.shape(self.seg_array))

            for i in range(sh[0]):
                for j in range(sh[1]):
                    for k in range(sh[2]):

                        if not self.seg_array[k,i,j] == self.vol_id: # note transpose
                            pRB[i,j,k] = 0
                            continue

                        rDsq = sweep_radius ** 2

                        xsq = x[i,j,k] ** 2
                        ysq = y[i,j,k] ** 2
                        zsq = z[i,j,k] ** 2

                        tworDz = 2 * sweep_radius * z[i,j,k]
                        tmpone = rDsq + ysq + zsq + tworDz
                        tmptwo = xsq + ysq + zsq + tworDz + 2 * rDsq
                        tmpthree = np.sqrt(tmpone) * 2 * sweep_radius

                        pRB[i,j,k] = np.sqrt(tmptwo - tmpthree)

                print(str(int(100*i/sh[0])),end='\r')
            # Ignore these...
            pBAngle = 0
            pVAngle = 0


        self.vrbs("There are some non.inf's: "+str(np.sum(pRB<np.inf)))

        return pBAngle, pVAngle, pRB
        # return 0, 0, int(np.random.random()*100)

    def do_std_regress(self,depth_values,std_values, weights=None):
        n = len(depth_values)

        # clean nan's
        mask = np.isfinite(depth_values) & np.isfinite(std_values)

        if weights == None:
            a, b = np.polyfit([depth_values[i] for i in range(n) if mask[i]],
                            [std_values[i] for i in range(n) if mask[i]],
                                1)
        else:
            a, b = np.polyfit([depth_values[i] for i in range(n) if mask[i]],
                            [std_values[i] for i in range(n) if mask[i]],
                            1,
                            w = [weights[i] for i in range(n) if mask[i]])

        return a*np.array(depth_values) + b

    # --- UTILITIES
    def get_distance_map(self):
        '''
            get_distance_map
        Get the distance information of voxels from the transducer head as a multi-dimensional array.
        '''
        if not self.kretz_supplied:
            raise Exception("Kretz data has not been supplied. Use load_kretz.")
        
        self.vrbs("Building distance map...")

        bmode_spacing = np.frombuffer(self.kretz_metadata.getDataFromTags(
        int('0xc100', 16), int('0x201', 16)), np.float64)[0] * 1000.0
        volume_offset = np.frombuffer(self.kretz_metadata.getDataFromTags(
                int('0xc200', 16), int('0x202', 16)), np.float64)[0]
        sweep_radius = bmode_spacing * volume_offset

        # dimensions = np.shape(np.transpose(self.pd_array, (0, 2, 1))) # not efficient!
        dimensions = self.pd_img.GetSize() # (!) Do we have to use _img here? Can we get this info. from _array?
        corner1 = self.pd_img.TransformIndexToPhysicalPoint((0,0,0))
        corner2 = self.pd_img.TransformIndexToPhysicalPoint((dimensions[0]-1,dimensions[1]-1,dimensions[2]-1))

        # print(dimensions)
        # print(np.shape(self.pd_array))
        # print(np.shape(self.pd_array)[2, 0, 1])

        dimensions_pd = np.shape(self.pd_array)

        # if not self.zoom == 1:
        #     dimensions = np.shape

        x = np.linspace(corner1[0], corner2[0], dimensions_pd[2])
        y = np.linspace(corner1[1], corner2[1], dimensions_pd[1])
        z = np.linspace(corner1[2], corner2[2], dimensions_pd[0])
        self.vrbs("Creating meshgrid...")

        # if not self.zoom == 1: # SOMETHING GOING ON HERE!!!
        #     x = ndimage.zoom(x, (self.zoom, self.zoom, self.zoom))
        #     y = ndimage.zoom(y, (self.zoom, self.zoom, self.zoom))
        #     z = ndimage.zoom(z, (self.zoom, self.zoom, self.zoom))

        xx, yy, zz = np.meshgrid(x, y, z)

        self.vrbs("Get radial distances...")
        _, _, rDist = self.get_angle(xx, yy, zz, sweep_radius)
        distance = np.transpose(rDist, (2, 0, 1))

        return distance
    
    def vrbs(self, msg):
        ''' 
                vrbs
            If self.verbose is True, print message.
        '''    
        if self.verbose:
            try:
                print('[verbose] ' + msg)
            except(TypeError):
                try:
                    print('[verbose]... ')
                    print(msg)
                except:
                    print('[verbose] Trouble printing verbose message.')
