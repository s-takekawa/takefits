from astropy.wcs import WCS
from astropy import units as u
from astropy.constants import c

class FreqToVelocity:
    def __init__(self, header):
        self.header = header.copy()
        try:
            self.wcs = WCS(header)
        except Exception as e:
            #print(f"Warning: WCS initialization failed in __init__: {e}")
            self.wcs = None
        self.freq_axis = None
        self.converted = False
        self.c_speed = c.to('m/s').value
        self.original_axis_type = None
        self.original_axis_unit = None
        self.frequency_unit_before_conversion = None
        self._find_frequency_axis()
        self.restfreq = None


    def _find_frequency_axis(self):
        naxis = self.header['NAXIS']
        for i in range(1, naxis + 1):
            ctype = self.header.get(f'CTYPE{i}', '').upper()
            if 'FREQ' in ctype:
                self.freq_axis = i
                self.original_axis_type = self.header.get(f'CTYPE{i}', '')
                self.original_axis_unit = self.header.get(f'CUNIT{i}', '').strip()
                break
        if self.freq_axis is not None:
            self._confirm_conversion()


    def _confirm_conversion(self):
        yellow = "\033[93m"
        cyan = "\033[96m"
        reset = "\033[0m"
        
        print(f'{cyan}The {self.ordinal(self.freq_axis)} axis is in Frequency.\n{yellow}Trying conversion to radial velocity... {reset}')
        #response = input(f'{cyan}The {self.ordinal(self.freq_axis)} axis is in Frequency.\n{yellow}  Do you want to convert to radial velocity? (y/n): \n {reset}').strip().lower()
        #if response == 'n' or response == 'no' :
        #    print(f'{cyan}Frequency conversion skipped.{reset}')
        #    self.freq_axis = None
        #    self.converted = False 
        #else:
        if True:
            try:
                self._get_rest_frequency()
            except ValueError:
                print(f" The RESTFREQ key does not exist,\n  so the frequency-to-velocity conversion is skipped.")
                return
            self._convert_units_to_Hz()
            self.convert_to_velocity()
            self.converted = True
            print(f"{cyan}Converted frequency to radial velocity.{reset}")

    def ordinal(self, n: int):
        if 11 <= (n % 100) <= 13:
            suffix = 'th'
        else:
            suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
        return str(n) + suffix

    def _get_rest_frequency(self):
        self.restfreq = self.header.get('RESTFRQ', self.header.get('RESTFREQ', None))
        if self.restfreq is None:
            raise ValueError('Rest frequency (RESTFRQ or RESTFREQ) does not exsist in the header')
        self.restfreq = float(self.restfreq)


    def _convert_units_to_Hz(self):
        self.crval_freq = self.header[f'CRVAL{self.freq_axis}']
        self.cdelt_freq = self.header[f'CDELT{self.freq_axis}']
        self.crpix_freq = self.header[f'CRPIX{self.freq_axis}']
        self.cunit_freq = self.header.get(f'CUNIT{self.freq_axis}', 'Hz')

        self.frequency_unit_before_conversion = self.cunit_freq

        if self.cunit_freq != 'Hz':
            self.crval_freq = (self.crval_freq * u.Unit(self.cunit_freq)).to(u.Hz).value
            self.cdelt_freq = (self.cdelt_freq * u.Unit(self.cunit_freq)).to(u.Hz).value
            self.restfreq = (self.restfreq * u.Unit('Hz')).to(u.Hz).value

    def convert_to_velocity(self):
        if self.freq_axis is None:
            return

        crval_vel = self.c_speed * (self.restfreq - self.crval_freq) / self.restfreq

        cdelt_vel = - (self.c_speed * self.cdelt_freq) / self.restfreq

        self.header[f'CRVAL{self.freq_axis}'] = crval_vel
        self.header[f'CDELT{self.freq_axis}'] = cdelt_vel
        self.header[f'CTYPE{self.freq_axis}'] = 'VRAD'
        self.header[f'CUNIT{self.freq_axis}'] = 'm/s'

        try:
            self.wcs = WCS(self.header)
        except Exception as e:
            print(f"Warning: WCS re-initialization failed in convert_to_velocity: {e}")
            self.wcs = None
