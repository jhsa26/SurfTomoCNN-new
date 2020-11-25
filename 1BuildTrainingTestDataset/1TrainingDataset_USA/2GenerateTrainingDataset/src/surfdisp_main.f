c calculate surface wave dispersion curve for a given model, given mode, given
c velocity type, and given periods
c by Huajian Yao: MIT hjyao@mit.edu
c last modified: for output dispersion data to file
c                08/25/2010 at IGPP/SIO UCSD
c                
       program surfdisp_main
	 
        parameter(LER=0,LIN=5,LOT=6)	 
        integer NL, NL2, NLAY
        parameter(NL=800,NLAY=800,NL2=NL+NL)
        integer NP
        parameter (NP=512)

c-----
c     LIN - unit for FORTRAN read from terminal
c     LOT - unit for FORTRAN write to terminal
c     LER - unit for FORTRAN error output to terminal
c     NL  - layers in model
c     NP  - number of unique periods
c-----
        double precision t(NP),cg(NP)
        character*50 mdlfile, disperfile

		integer*4 iwave, mode, igr, kmax
		

c-----
c     common for iget
c-----
        common/isomod/dl(NLAY),va(NLAY),vb(NLAY),rrho(NLAY),
     1      qa(NLAY),qb(NLAY),etap(NLAY),etas(NLAY), 
     2      frefp(NLAY), frefs(NLAY), refdep(NLAY)
        integer mmax, iunit, iiso, iflsph, idimen, icnvel
        real*4 thkm(NLAY), vpm(NLAY), vsm(NLAY), rhom(NLAY)
        character title*80
c-----
c     machine dependent initialization
c-----
c        call mchdep()
c-----

c        open(3,file='disp96.in')
c        rewind 3
        read(*,*) mdlfile
        read(*,*) disperfile
		print *, mdlfile, disperfile
		read(*,*) iwave, mode, igr, kmax
c		print *, iwave, mode-1, igr, kmax
		read(*,*) (t(i),i=1,kmax)
c		write(*,*)(sngl(t(i)),i=1,kmax)

c-----
c     get velocity model
c-----
        call getmod(2,mdlfile,mmax,title,iunit,iiso,iflsph,
     1      idimen,icnvel,ierr,.true.)

c-----
c     save model to new parameters
c-----
        do 39 i=1,mmax
            vsm(i) = vb(i)
            vpm(i) = va(i)
            thkm(i) = dl(i)
            rhom(i) = rrho(i)
   39   continue	 
c-----
c     calculate dispersion curve for given wave type, mode, and periods
c-----	 
	     call surfdisp96(thkm,vpm,vsm,rhom,mmax,iflsph,iwave,mode,
     &                   igr,kmax,t,cg)
         print *, '--- Dispersion Data ---'		 
         open(5,file=disperfile)
		 do i=1,kmax
		    print *, sngl(t(i)),sngl(cg(i))
		    if(cg(i).gt.0.0001) then
		        write(5,*) sngl(t(i)),sngl(cg(i))
		    endif
         enddo
         close(5)
		   
        stop
        end		
	 
	 
c
      subroutine mchdep()
c---------------------------------------------------------------------c
c                                                                     c
c      COMPUTER PROGRAMS IN SEISMOLOGY                                c
c      VOLUME V                                                       c
c                                                                     c
c      PROGRAM: mchdep                                                c
c                                                                     c
c      COPYRIGHT 1996 R. B. Herrmann                                  c
c                                                                     c
c      Department of Earth and Atmospheric Sciences                   c
c      Saint Louis University                                         c
c      221 North Grand Boulevard                                      c
c      St. Louis, Missouri 63103                                      c
c      U. S. A.                                                       c
c                                                                     c
c---------------------------------------------------------------------c
        implicit none
      return
      end


        subroutine getmod(rlun,mname,mmax,title,iunit,iiso,iflsph,
     1      idimen,icnvel,ierr,listmd)		
c-----
c       HISTORY
c
c       09 08 2000  gave ierr an initial default value for g77
c       01 13 2001  put in close(lun) if file is not model file
c       03 MAY 2002     Modify to permit read from standard input
c       06 JUL 2005 moved inquire to permit use of STDIN
c
c-----
c       General purpose model input
c       This model specification is designed to be as 
c           general as possible
c
c       Input lines
c       Line 01: MODEL
c       Line 02: Model Name
c       Line 03: ISOTROPIC or ANISOTROPIC or 
c           TRANSVERSELY ANISOTROPIC
c       Line 04: Model Units, First character 
c           is length (k for kilometer
c           second is mass (g for gm/cc), third is time (s for time)
c       Line 05: FLAT EARTH or SPHERICAL EARTH
c       Line 06: 1-D, 2-D or 3-D
c       Line 07: CONSTANT VELOCITY
c       Line 08: open for future use
c       Line 09: open for future use
c       Line 10: open for future use
c       Line 11: open for future use
c       Lines 12-end:   These are specific to the model
c           For ISOTROPIC the entries are
c           Layer Thickness, P-velocity, S-velocity, Density, Qp, Qs,
c           Eta-P, Eta S (Eta is frequency dependence), 
c           FreqRefP, FreqRefP
c-----
cMODEL
cTEST MODEL.01
cISOTROPIC
cKGS
cFLAT EARTH
c1-D
cCONSTANT VELOCITY
cLINE08
cLINE09
cLINE10
cLINE11
c H  VP  VS   RHO   QP  QS   ETAP   ETAS REFP  REFS
c1.0    5.0 3.0 2.5 0.0 0.0 0.0 0.0 1.0 1.0
c2.0    5.1 3.1 2.6 0.0 0.0 0.0 0.0 1.0 1.0
c7.0    6.0 3.5 2.8 0.0 0.0 0.0 0.0 1.0 1.0
c10.0   6.5 3.8 2.9 0.0 0.0 0.0 0.0 1.0 1.0
c20.0   7.0 4.0 3.0 0.0 0.0 0.0 0.0 1.0 1.0
c40.0   8.0 4.7 3.3 0.0 0.0 0.0 0.0 1.0 1.0
c-----
c-----
c       rlun    I*4 - logical unit for reading model file. This
c                 unit is released after the use of this routine
c       mname   C*(*)   - model name - if this is stdin or 
c           STDIN just read
c                 from standard input
c       mmax    I*4 - number of layers in the model, last layer is
c                    halfspace
c       title   C*(*)   - title of the model file
c       iunit   I*4 - 0 Kilometer, Gram, Sec
c       iiso    I*4 - 0 isotropic 
c                 1 transversely anisotropic 
c                 2 general anisotropic 
c       iflsph  I*4 - 0 flat earth model
c                 1 spherical earth model
c       idimen  I*4 - 1 1-D
c               - 2 2-D
c               - 3 3-D
c       icnvel  I*4 - 0 constant velocity
c                 1 variable velocity
c       ierr    I*4 - 0 model file correctly read in
c               - -1 file does not exist
c               - -2 file is not a model file
c                 -3 error in the model file
c       listmd  L   - .true. list the model
c------

        implicit none
        character mname*(*), title*(*)
        integer rlun
        integer*4 mmax, iunit, iiso, iflsph, idimen, icnvel
        integer*4 ierr
        character string*80
        logical listmd
c-----
c       LIN I*4 - logical unit for standard input
c       LOT I*4 - logical unit for standard output
c-----
        integer LIN, LOT, LER
        parameter (LIN=5,LOT=6,LER=0)

        integer NL
        parameter (NL=800)
        common/isomod/d(NL),a(NL),b(NL),rho(NL),
     1      qa(NL),qb(NL),etap(NL),etas(NL), 
     2      frefp(NL),frefs(NL),refdep(NL)
        real d,a,b,rho,qa,qb,etap,etas,frefp,frefs,refdep
		
        logical ext
        character ftype*80
        integer lun, j, i, irefdp

c-----
c       test to see if the file exists
c-----
        ierr = 0
c-----
c       test for input
c-----
        if(MNAME(1:5).eq.'stdin' .or. mname(1:5).eq.'STDIN')then
c-----
c           do not open anything, use standard output
c-----
            lun = LIN
        else
            lun = rlun
            inquire(file=mname,exist=ext)
            if(.not.ext)then
                ierr = -1
                write(LER,*)'Model file does not exist'
                return
            endif
c-----
c           open the file
c-----
            open(lun,file=mname,status='old',form='formatted',
     1          access='sequential')
            rewind lun
        endif
c-----
c       verify the file type
c-----
c-----
c       LINE 01
c-----
        read(lun,'(a)')ftype
        if(ftype(1:5).ne.'model' .and. ftype(1:5).ne.'MODEL')then
            ierr = -2
            write(LER,*)'Model file is not in model format'
            close(lun)
            return
        endif
c-----
c       LINE 02
c-----
        read(lun,'(a)')title
c-----
c       LINE 03
c-----
        read(lun,'(a)')string
        if(string(1:3).eq.'ISO' .or. string(1:3).eq.'iso')then
            iiso = 0
        else if(string(1:3).eq.'TRA' .or. string(1:3).eq.'tra')then
            iiso = 1
        else if(string(1:3).eq.'ANI' .or. string(1:3).eq.'ani')then
            iiso = 2
        endif
c-----
c       LINE 04
c-----
        read(lun,'(a)')string
        if(string(1:3).eq.'KGS' .or. string(1:3).eq.'kgs')then
            iunit = 0
        endif
c-----
c       LINE 05
c-----
        read(lun,'(a)')string
        if(string(1:3).eq.'FLA' .or. string(1:3).eq.'fla')then
            iflsph = 0
        else if(string(1:3).eq.'SPH' .or. string(1:3).eq.'sph')then
            iflsph = 1
        endif
c-----
c       LINE 06
c-----
        read(lun,'(a)')string
        if(string(1:3).eq.'1-d' .or. string(1:3).eq.'1-D')then
            idimen = 1
        else if(string(1:3).eq.'2-d' .or. string(1:3).eq.'2-D')then
            idimen = 2
        else if(string(1:3).eq.'3-d' .or. string(1:3).eq.'3-D')then
            idimen = 3
        endif
c-----
c       LINE 07
c-----
        read(lun,'(a)')string
        if(string(1:3).eq.'CON' .or. string(1:3).eq.'con')then
            icnvel = 0
        else if(string(1:3).eq.'VAR' .or. string(1:3).eq.'var')then
            icnvel = 1
        endif
c-----
c       get lines 8 through 11
c-----
        do 900 i=8,11
            read(lun,'(a)')string
  900   continue
c-----
c       get model specifically for 1-D flat isotropic
c-----
c-----
c       get comment line
c-----
        read(lun,'(a)')string
        mmax = 0
        refdep(1) = 0.0
        irefdp = 0
        if(iiso.eq.0)then
 1000       continue
            j = mmax +1
                read(lun,*,err=9000,end=9000)d(j),a(j),b(j),
     1              rho(j),qa(j),qb(j),etap(j),etas(j),
     2              frefp(j),frefs(j)
                if(d(j).lt.0.0)then
                    d(j) = -d(j)
                    irefdp = j
                endif
                refdep(j+1) = refdep(j) + d(j)
            mmax = j
            go to 1000
 9000       continue
        endif
    1   format(' LAYER             H      P-VEL     S-VEL   DENSITY  ')
    2   format(' ',i5,5x,4f10.3)
    3   format(' ','-SURFACE ','- - - - - ','- - - - - ',
     1      '- - - - - ','- - - - - -')
        if(mmax.gt.0)then
            if(listmd)then
            ierr = 0
            write(LOT,1)
            do 2000 i=1,mmax
                write(LOT,2)
     1              i,d(i),a(i),b(i),rho(i)
                if(i.eq.irefdp)write(LOT,3)
 2000       continue
            endif
        else 
            ierr = -3
            write(LER,*)'Error in model file'
        endif
        if(lun.ne.LIN)close (lun)
		
c		do i=1,mmax
c		   write(*,*)refdep(i),b(i),d(i)
c		enddo
		
        return
        end	  
