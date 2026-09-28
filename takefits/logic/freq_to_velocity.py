import warnings

from astropy.wcs import WCS
from astropy import units as u
from astropy.constants import c

from takefits.core.spectral_units import (
    canonical_velocity_unit,
    header_axis_step,
    rest_frequency_hz,
    set_header_axis_step,
    velocity_convention,
)

class FreqToVelocity:
    to_frequency = False  # (RadioVelocityToFrequency's flag)

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
        self.restfreq = None  # set by the conversion (Hz)
        self._find_frequency_axis()


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
                print(
                    f"{yellow}No rest frequency (RESTFRQ / RESTFREQ): the axis stays in frequency.{reset}\n"
                    f"{cyan}  To show velocities, set RestFreq in Unit Conversion, save the cube and reopen it.{reset}"
                )
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
        step = header_axis_step(self.header, self.freq_axis)  # CDELT or CD matrix
        self.cdelt_freq = 1.0 if step is None else step
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
        set_header_axis_step(self.header, self.freq_axis, cdelt_vel)
        self.header[f'CTYPE{self.freq_axis}'] = 'VRAD'
        self.header[f'CUNIT{self.freq_axis}'] = 'm/s'

        try:
            self.wcs = WCS(self.header)
        except Exception as e:
            print(f"Warning: WCS re-initialization failed in convert_to_velocity: {e}")
            self.wcs = None


class RadioVelocityToFrequency:
    """The frequency axis a radio-velocity axis stands for (``load_fits(frequency_axis='frequency')``).

    f = f0 (1 - v / c) is linear in v, so CRVAL and the step (CDELT, or the
    CD row) map exactly and the data keep their order.  Optical and
    relativistic velocities are not linear in frequency and stay as they are,
    and so does an axis without a rest frequency or a velocity unit;
    ``reason`` then says why.  ``to_frequency`` is True when the axis was
    converted (``converted`` belongs to :class:`FreqToVelocity` and stays False).
    """

    converted = False

    def __init__(self, header, axis_number):
        self.header = header.copy()
        self.freq_axis = int(axis_number)  # the FITS axis, as FreqToVelocity names it
        self.to_frequency = False
        self.reason = None
        self.restfreq = None
        self.velocity_unit = None
        self.c_speed = c.to('m/s').value
        self._convert()

    def _convert(self):
        n = self.freq_axis
        header = self.header
        ctype = str(header.get(f'CTYPE{n}', '') or '').strip()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                wcs = WCS(header)
            translated = str(wcs.wcs.ctype[n - 1]).strip().upper()
            convention = velocity_convention(wcs, n - 1)
        except Exception:
            wcs, translated, convention = None, '', None
        if convention == 'radio' and translated != 'VRAD':
            convention = None  # a radio axis sampled non-linearly in frequency ('VRAD-xxx')
        if convention != 'radio':
            if convention in ('optical', 'relativistic'):
                self.reason = f"{convention} velocities ({ctype}) are not linear in frequency"
            else:
                self.reason = f"the velocity convention of CTYPE '{ctype}' is not known"
            return
        rest_hz = rest_frequency_hz(None, header)
        if rest_hz is None:
            self.reason = "there is no rest frequency (RESTFRQ / RESTFREQ)"
            return
        unit = canonical_velocity_unit(header.get(f'CUNIT{n}', ''))
        if unit is None:
            self.reason = f"CUNIT{n} does not say whether the velocities are in m/s or km/s"
            return
        to_ms = 1000.0 if unit == 'km/s' else 1.0
        step = header_axis_step(header, n)
        step_ms = (1.0 if step is None else float(step)) * to_ms  # FITS default CDELT 1
        crval_ms = float(header.get(f'CRVAL{n}', 0.0)) * to_ms
        header[f'CRVAL{n}'] = rest_hz * (1.0 - crval_ms / self.c_speed)
        set_header_axis_step(header, n, -rest_hz * step_ms / self.c_speed)
        header[f'CTYPE{n}'] = 'FREQ'
        header[f'CUNIT{n}'] = 'Hz'
        # The AIPS forms (VELO-LSR with VELREF) name the frame in CTYPE; keep it.
        specsys = str(getattr(getattr(wcs, 'wcs', None), 'specsys', '') or '').strip()
        if 'SPECSYS' not in header and specsys:
            header['SPECSYS'] = specsys
        self.restfreq = rest_hz
        self.velocity_unit = unit
        self.to_frequency = True
